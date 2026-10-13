"""Registered exploratory analysis of sealed 3x20 panels; never simulates."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from drl.paper_recovery.receipts import atomic_write_json, canonical_hash, sha256_file

GROUPS = {
 'A0_full':('A0 full / 657111','A0 full / 657112','A0 full / 657113'),
 'A1_no_history':('A1 no history / 657121','A1 no history / 657122','A1 no history / 657123'),
 'A2_no_admission':('A2 no admission / 657131','A2 no admission / 657132','A2 no admission / 657133'),
 'A3_no_security_critic':('A3 no security critic / 657141','A3 no security critic / 657142','A3 no security critic / 657143'),
}

def exact_sign_flip(differences):
    d=np.asarray(differences,dtype=float)
    if d.ndim!=1 or not np.isfinite(d).all() or not 1<=len(d)<=20:
        raise ValueError('Invalid paired environment differences')
    n=len(d); assignments=1<<n
    observed=abs(float(d.mean()))
    count=0
    for start in range(0,assignments,8192):
        indices=np.arange(start,min(start+8192,assignments),dtype=np.uint64)
        signs=2*((indices[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(float)-1
        count+=int(np.count_nonzero(np.abs(signs@d/n)>=observed-1e-12))
    return count/assignments

def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p)
    adjusted=np.empty_like(p)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted.tolist()

def bootstrap_contrast(left,right,rng,reps):
    # Training replicates remain independent within arm; scenario resampling paired.
    left_indices=rng.integers(0,3,size=(reps,3))
    right_indices=rng.integers(0,3,size=(reps,3))
    environments=rng.integers(0,20,size=(reps,20))
    per_environment=left[left_indices].mean(axis=1)-right[right_indices].mean(axis=1)
    values=np.take_along_axis(per_environment,environments,axis=1).mean(axis=1)
    return np.quantile(values,[0.025,0.975]).tolist()

def compute_campaign(modes,plan):
    seeds=plan['environment_seeds']
    if seeds!=list(range(657201,657221)) or plan['hierarchical_bootstrap_replicates']!=10000:
        raise ValueError('Registered analysis plan drift')
    arrays={arm:np.asarray([[modes[name]['per_seed'][str(s)]['mean_tps_late_1500_1999'] for s in seeds] for name in names],dtype=float) for arm,names in GROUPS.items()}
    if any(a.shape!=(3,20) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Exactly3 independent checkpoints x20 paired environments required')
    rng=np.random.default_rng(plan['analysis_rng_seed'])
    contrasts=[]
    for arm in ('A1_no_history','A2_no_admission','A3_no_security_critic'):
        environment_differences=arrays[arm].mean(axis=0)-arrays['A0_full'].mean(axis=0)
        interval=bootstrap_contrast(arrays[arm],arrays['A0_full'],rng,10000)
        contrasts.append({'contrast':arm+' minus A0_full','late_tps_difference':float(environment_differences.mean()),'hierarchical_percentile95_lower':interval[0],'hierarchical_percentile95_upper':interval[1],'conditional_environment_exact_signflip_p':exact_sign_flip(environment_differences),'paired_environment_differences':environment_differences.tolist()})
    for row,p in zip(contrasts,holm_adjust([d['conditional_environment_exact_signflip_p'] for d in contrasts])):
        row['conditional_environment_Holm_p']=p
    arm_summary={arm:{'checkpoint_labels':list(GROUPS[arm]),'individual_checkpoint_late_tps':a.mean(axis=1).tolist(),'equal_weight_late_tps':float(a.mean()),'between_training_seed_sd_late_tps':float(a.mean(axis=1).std(ddof=1))} for arm,a in arrays.items()}
    checks={
        'all_four_arms_exact_three_checkpoints_twenty_paired_environments':all(a.shape==(3,20) for a in arrays.values()),
        'finite_complete_panels_no_seed_filtering':all(np.isfinite(a).all() for a in arrays.values()),
        'equal_weight_means_match_sealed_aggregates':all(np.isclose(arm_summary[arm]['equal_weight_late_tps'],np.mean([modes[name]['aggregate']['mean_tps_late_1500_1999'] for name in names])) for arm,names in GROUPS.items()),
        'registered_three_primary_contrasts_only': [r['contrast'] for r in contrasts]==plan['primary_contrasts'],
        'bootstrap_10000_exact_signflip_Holm_finite':all(np.isfinite([r['late_tps_difference'],r['hierarchical_percentile95_lower'],r['hierarchical_percentile95_upper'],r['conditional_environment_exact_signflip_p'],r['conditional_environment_Holm_p']]).all() and 0<=r['conditional_environment_exact_signflip_p']<=r['conditional_environment_Holm_p']<=1 for r in contrasts),
    }
    if not all(checks.values()): raise RuntimeError(checks)
    return {'schema':'qrelay-batch11-exploratory-campaign-analysis-1','status':'PASS','checks':checks,'training_checkpoints_per_arm':3,'environment_scenarios':20,'bootstrap_replicates':10000,'analysis_rng_seed':plan['analysis_rng_seed'],'arm_summary':arm_summary,'primary_contrasts':contrasts,'interpretation':plan['interpretation'],'training_replicate_unit':'independent training checkpoint (3 per arm), NOT scenario/trajectory','shared_panel_previously_observed':True,'secondary_significance_tests':False}

def analyze_and_write(input_path,output):
    input_path,output=Path(input_path),Path(output)
    root=Path(__file__).resolve().parents[1]
    plan_path=root/'revision_R1_results/03_ablation/monitor_batch_11/analysis_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    doc=json.loads(input_path.read_text(encoding='utf-8'))
    modes={'A1 no history / 657123':doc['deterministic_primary'],**doc['reference_ppo_modes']}
    result=compute_campaign(modes,plan)
    result.update(analysis_plan_sha256=sha256_file(plan_path),evaluation_sha256=sha256_file(input_path),analysis_source_sha256=sha256_file(Path(__file__)),reference_hashes=doc['reference_hashes'])
    for name in ('campaign_analysis.json','campaign_primary_contrasts.csv'):
        if (output/name).exists(): raise RuntimeError('Preserve existing analysis output')
    result['receipt_hash']=canonical_hash(result)
    atomic_write_json(output/'campaign_analysis.json',result)
    with (output/'campaign_primary_contrasts.csv').open('x',encoding='utf-8',newline='') as stream:
        fields=[k for k in result['primary_contrasts'][0] if k!='paired_environment_differences']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows({k:r[k] for k in fields} for r in result['primary_contrasts'])
    text=['','## Registered exploratory three-checkpoint analysis','',
          'Hierarchy:3 independent training checkpoints per arm,20 shared paired environment scenarios. Bootstrap10,000 replicates resamples training checkpoints within arms and shared scenarios; percentile95 intervals. Exact two-sided sign flip enumerates all2^20 assignments on training-seed-averaged scenario differences; Holm adjusts ONLY the three primary lateTPS contrasts. Scenario p-values condition on these checkpoints, not a population of training seeds. Extension was planned after observing the original panel; no unseen-panel qualification or guaranteed/population significance claim.','',
          '| Contrast | Difference TPS | Hierarchical95% interval | Conditional scenario p | Holm p |',
          '|---|---:|---:|---:|---:|']
    for r in result['primary_contrasts']:
        text.append(f"| {r['contrast']} | {r['late_tps_difference']:.3f} | [{r['hierarchical_percentile95_lower']:.3f}, {r['hierarchical_percentile95_upper']:.3f}] | {r['conditional_environment_exact_signflip_p']:.6f} | {r['conditional_environment_Holm_p']:.6f} |")
    text+=['','Between-training-seed variability (SD of checkpoint mean lateTPS, n=3):']
    for arm,d in result['arm_summary'].items():
        text.append(f"- {arm}: mean{d['equal_weight_late_tps']:.3f} TPS; between-seed SD{d['between_training_seed_sd_late_tps']:.3f}.")
    return text
