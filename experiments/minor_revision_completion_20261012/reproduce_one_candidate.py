"""Once-only candidate reproducibility smoke, not another formal trajectory."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from experiments.paper_recovery_security_speed_v5.evaluate_disaster import _load_policy,_evaluate_one,PPO_NAME
from drl.paper_recovery_security_v4.config import security_v4_config
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((ROOT/'experiments/paper_recovery_security_speed_v5/disaster_protocol.json').read_text())
    policy,receipt=_load_policy(protocol) # No environment constructed in this qualification.
    parent=json.loads((ROOT/'results/paper_recovery_security_speed_v5/heldout_disaster_seed_657111/evaluation.json').read_text())
    if '--execute' not in sys.argv:print('Qualified imports, exact source/config/checkpoint closure; no episode executed');return
    out=ROOT/'candidate_reproduction_smoke';out.mkdir(exist_ok=True)
    with (out/'once_only_claim.json').open('x') as f:json.dump({'scope':'one candidate smoke, not formal evaluation','seed':657201,'checkpoint_sha256':receipt['checkpoint_sha256'],'no_automatic_retry':True},f)
    try:
        row,timeline,exo=_evaluate_one(PPO_NAME,657201,security_v4_config(training=False),policy,0)
        expected=parent['panel']['deterministic_primary']['per_seed']['657201']
        assert len(timeline)==2000 and all(math.isfinite(x) for x in timeline)
        assert abs(row['mean_tps_late_1500_1999']-expected['mean_tps_late_1500_1999'])<1e-8
        assert exo==parent['panel']['paired_exogenous_hashes']['657201']
        assert row['forced_zero_tps_steps']==0
        result={'status':'PASS','seed':657201,'steps':2000,'expected_late_tps':expected['mean_tps_late_1500_1999'],'actual_late_tps':row['mean_tps_late_1500_1999'],'paired_exogenous_hash':exo,'checkpoint_sha256':receipt['checkpoint_sha256'],'source_hash':receipt['source_hash'],'runtime':{'python':sys.version,'numpy':__import__('numpy').__version__,'torch':torch.__version__},'formal_campaign_untouched':True,'scope':'Candidate overlaid on a fresh old public checkout; not a public release download and not new statistical evidence'}
        (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    except BaseException as e:
        (out/'failure.json').write_text(json.dumps({'status':'terminal_failure_no_retry','error':repr(e)},indent=2)+'\n');raise
if __name__=='__main__':main()
