"""Render the sealed batch11 comparison; this module never runs the simulator."""
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drl.paper_recovery.receipts import canonical_hash, sha256_file

NEW = "A1 no history / 657123"
THIRD_A3 = "A3 no security critic / 657143"
THIRD_A2 = "A2 no admission / 657133"
THIRD_A0 = "A0 full / 657113"
SECOND_A1 = "A1 no history / 657122"
FIRST_A1 = "A1 no history / 657121"
SECOND_A3 = "A3 no security critic / 657142"
FIRST_A3 = "A3 no security critic / 657141"
A0_1, A0_2 = "A0 full / 657111", "A0 full / 657112"
A2_1, A2_2 = "A2 no admission / 657131", "A2 no admission / 657132"
COLORS = {
    A0_1: "#075cbe", A0_2: "#6b4ab5", A2_1: "#ce8400", A2_2: "#bf402d",
    FIRST_A3: "#153c34", SECOND_A3: "#65a58e", NEW: "#0f798a", THIRD_A3: "#237643", THIRD_A2: "#a05e00", THIRD_A0: "#1734a6", SECOND_A1: "#b122ad", FIRST_A1: "#e39dde", "CMAB": "#007f82", "GBA": "#259c5e",
    "SA": "#bc649d", "FMG": "#c67b07", "OCD-inspired": "#8f6477",
    "Risk-Prior": "#9a93a3", "Random": "#68737b", "Static-with-valid-refill": "#b96568",
}
DISPLAY = {
    "OCD-inspired": "OCD", "Static-with-valid-refill": "Static",
    A0_1: "A0 / 657111", A0_2: "A0 / 657112", A2_1: "A2 / 657131",
    A2_2: "A2 / 657132", FIRST_A3: "A3 / 657141", SECOND_A3: "A3 / 657142", NEW: "A1 / 657123", THIRD_A3: "A3 / 657143", THIRD_A2: "A2 / 657133", THIRD_A0: "A0 / 657113", SECOND_A1: "A1 / 657122", FIRST_A1: "A1 / 657121",
}


def smooth(values):
    values = np.asarray(values, dtype=float)
    result = values.copy()
    for start, end in ((0, 1000), (1000, 2000)):
        cumulative = np.r_[0, np.cumsum(values[start:end])]
        stop = np.arange(1, end - start + 1)
        begin = np.maximum(0, stop - 25)
        result[start:end] = (cumulative[stop] - cumulative[begin]) / (stop - begin)
    return result


def render(input_path: Path, output: Path) -> None:
    input_path, output = Path(input_path), Path(output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validation = json.loads(input_path.with_name("validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("evaluation_sha256") != sha256_file(input_path):
        raise RuntimeError("Refuse to render unvalidated or changed evaluation data")
    if canonical_hash({k: v for k, v in payload.items() if k != "receipt_hash"}) != payload.get("receipt_hash"):
        raise RuntimeError("Evaluation receipt hash mismatch")

    modes = {NEW: payload["deterministic_primary"], **payload["reference_ppo_modes"]}
    aggregates = {name: mode["aggregate"] for name, mode in modes.items()} | payload["baselines"]["aggregates"]
    guards = {name: mode["security_guard"] for name, mode in modes.items()} | payload["baselines"]["security_guards"]
    timelines = {name: mode["mean_tps_timeline"] for name, mode in modes.items()} | payload["baselines"]["mean_tps_timeline"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(17, 13.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 1))
    main = fig.add_subplot(grid[0, :])
    for low, high, color in ((1000, 1200, "#fae0df"), (1200, 1500, "#faefdb"), (1500, 2000, "#edf4fb")):
        main.axvspan(low, high, color=color, alpha=.65)
    order = [name for name in timelines if name not in modes] + [A0_1, A0_2, THIRD_A0, A2_1, A2_2, THIRD_A2, FIRST_A3, SECOND_A3, THIRD_A3, FIRST_A1, SECOND_A1, NEW]
    for name in order:
        highlight = name in modes
        main.plot(smooth(timelines[name]), color=COLORS[name], label=DISPLAY.get(name, name),
                  lw=2.5 if highlight else 1.05, alpha=1 if highlight else .62,
                  ls="--" if name in (A0_2, A2_1, FIRST_A3, FIRST_A1) else "-", zorder=6 if highlight else 2)
    main.axvline(1000, color="#a42c38", ls="--", lw=1.4)
    main.set(xlim=(0, 1999), ylim=(0, None), xlabel="Control step", ylabel="Delivered service throughput (TPS)")
    main.set_title("(a) Third A1 no-history checkpoint vs A0, A2, A3 and baselines - 20 paired disaster scenarios", loc="left", pad=93)
    main.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.03), frameon=False, fontsize=8.8)
    main.text(.51, .97, "30% permanent node loss at step 1000 + cascading stress", transform=main.transAxes,
              va="top", color="#982332", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92})
    main.text(.01, .97, "Trailing 25-step means; no smoothing across the event", transform=main.transAxes,
              va="top", fontsize=9, color="#444444", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    main.grid(alpha=.2)

    left, right = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    names = sorted(aggregates, key=lambda name: aggregates[name]["mean_tps_late_1500_1999"])
    y = np.arange(len(names))
    late = [aggregates[name]["mean_tps_late_1500_1999"] for name in names]
    zero = [aggregates[name]["zero_tps_steps_post_event"] / 10 for name in names]
    left.barh(y, late, color=[COLORS[name] for name in names], height=.64)
    right.barh(y, zero, color=[COLORS[name] for name in names], height=.64)
    for i, name in enumerate(names):
        left.text(late[i] + 3, i, f"{late[i]:.1f}", va="center", fontsize=8.8)
        guard = guards[name]
        right.text(zero[i] + .6, i, f"{zero[i]:.2f}% | max={guard['global_max_actual_security_exposure']} " + ("PASS" if guard["strict_guard_pass"] else "FAIL"),
                   va="center", fontsize=7.8, color="#1e694a" if guard["strict_guard_pass"] else "#a12535")
    labels = [DISPLAY.get(name, name) for name in names]
    left.set(yticks=y, yticklabels=labels, xlim=(0, max(late) * 1.2), xlabel="Mean TPS, steps 1500-1999")
    right.set(yticks=y, yticklabels=labels, xlim=(0, max(55, max(zero) * 1.52)), xlabel="Post-event zero-service steps (%)")
    left.set_title("(b) Late-window delivered service", loc="left")
    right.set_title("(c) Availability and actual-exposure guard (f = 6)", loc="left")
    left.grid(axis="x", alpha=.2)
    right.grid(axis="x", alpha=.2)
    for name in ("disaster_a1_comparison.png", "disaster_a1_comparison.pdf", "baseline_comparison.csv", "REPORT.md"):
        if (output / name).exists():
            raise RuntimeError(f"Preserve existing rendering: {name}")
    fig.savefig(output / "disaster_a1_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "disaster_a1_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name in names[::-1]:
        aggregate, guard = aggregates[name], guards[name]
        rows.append({"method": name, "late_tps": aggregate["mean_tps_late_1500_1999"],
                     "pre_tps": aggregate["mean_tps_pre_0_999"],
                     "failure_pct": aggregate["failure_rate_post_1000_1999"] * 100,
                     "zero_service_pct": aggregate["zero_tps_steps_post_event"] / 10,
                     "max_actual_exposure": guard["global_max_actual_security_exposure"],
                     "violation_pct": guard["post_event_violation_fraction"] * 100,
                     "guard_pass": guard["strict_guard_pass"]})
    with (output / "baseline_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "# Batch 11: A1 no history - seed 657123", "",
        "Same registered 20 scenarios and 2000 steps. Three A0, three A2, three A3 and both previous A1 checkpoints, and eight baseline panels were reused, not rerun.", "",
        "| Method | Late TPS | Zero service (%) | Failure (%) | Max exposure | Violation (%) | Guard f=6 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(f"| {row['method']} | {row['late_tps']:.2f} | {row['zero_service_pct']:.2f} | {row['failure_pct']:.2f} | {row['max_actual_exposure']} | {row['violation_pct']:.3f} | {'PASS' if row['guard_pass'] else 'FAIL'} |")
    best = max(payload["baselines"]["aggregates"], key=lambda name: aggregates[name]["mean_tps_late_1500_1999"])
    table += ["", "## Conditional paired scenario comparisons", "",
              "Fixed checkpoints; 95% paired t intervals over20shared scenarios, unadjusted. Intervals are conditional on fixed checkpoints, not population training-seed intervals.", ""]
    for name in (A0_1, A0_2, THIRD_A0, A2_1, A2_2, THIRD_A2, FIRST_A3, SECOND_A3, THIRD_A3, FIRST_A1, SECOND_A1, best):
        interval = payload["comparisons"][name]["late_tps_difference"]
        table.append(f"- A1/657123 minus {name}: {interval['mean']:.2f} TPS, 95% CI [{interval['ci95_lower']:.2f}, {interval['ci95_upper']:.2f}].")

    groups = (("A1 no history", (FIRST_A1, SECOND_A1, NEW), "657121,657122,657123"),
              ("A0 full", (A0_1, A0_2, THIRD_A0), "657111,657112,657113"),
              ("A2 no admission", (A2_1, A2_2, THIRD_A2), "657131,657132,657133"),
              ("A3 no security critic", (FIRST_A3, SECOND_A3, THIRD_A3), "657141,657142,657143"))
    table += ["", "## Equal-weight descriptive means across available training checkpoints", "",
              "| Arm | Training seeds | Late TPS | Zero service (%) | Failure (%) | Maximum actual exposure | Violation (%) |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, pair, seed_text in groups:
        mean_late = np.mean([aggregates[name]["mean_tps_late_1500_1999"] for name in pair])
        mean_zero = np.mean([aggregates[name]["zero_tps_steps_post_event"] / 10 for name in pair])
        max_exposure = max(guards[name]["global_max_actual_security_exposure"] for name in pair)
        mean_failure = np.mean([aggregates[name]["failure_rate_post_1000_1999"] * 100 for name in pair])
        mean_violation = np.mean([guards[name]["post_event_violation_fraction"] * 100 for name in pair])
        table.append(f"| {label} | {seed_text} | {mean_late:.2f} | {mean_zero:.2f} | {mean_failure:.2f} | {max_exposure} | {mean_violation:.3f} |")
    table += ["", "All four arms have THREE independent training checkpoints each. The registered extension training/evaluation campaign is complete only after independent evidence and visual delivery validation. This extension is exploratory and uses an already observed shared panel; no unseen-panel or population-significance claim. The same 20 scenarios are paired outcomes, not independent training replicates; arm means are descriptive and paired intervals condition on fixed checkpoints.", "",
              "Validation PASS establishes evidence integrity, not security-guard success. Preserve negative outcomes. Training step/s is not held-out disaster TPS."]
    from analysis.analyze_minor_revision_batch11 import analyze_and_write
    table += analyze_and_write(input_path, output)
    (output / "REPORT.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sealed batch-11 comparison only; never runs the simulator.")
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.evaluation, args.output)


if __name__ == "__main__":
    main()



