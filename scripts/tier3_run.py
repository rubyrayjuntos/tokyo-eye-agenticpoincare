"""Tier 3: Matched-pair differential analysis."""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATASET_CONFIG_FILE', 'tokyoeyes_dataset_config.json')

from data_science.sub_agents.dtie.tools import run_dtie_pipeline

pairs = [
    ('4OBE', '4LDJ', 'KRAS_WT_vs_G12C'),
    ('4DSO', '4TQ9', 'KRAS_G12D_vs_G12V'),
    ('3TV4', '4MNE', 'BRAF_WT_vs_V600E'),
    ('1M17', '2ITV', 'EGFR_WT_vs_L858R'),
]

results = {}
for gdp, gtp, name in pairs:
    print(f'Running {name}: {gdp} vs {gtp}...', flush=True)
    result = run_dtie_pipeline(
        gdp_pdb_id=gdp, gtp_pdb_id=gtp,
        pipeline_mode='source_leak_v4',
        n_landmarks=200, effector_sites='',
        checkpoint='scientific',
    )
    data = json.loads(result)
    results[name] = data
    status = data['status']
    if status == 'ok':
        p2 = data.get('phase2', {})
        p4b = data.get('phase4b', {})
        print(f'  {name}: {p2.get("n_doorways")} doorways, L2={p4b.get("lambda_2_gdp",0):.4f}/{p4b.get("lambda_2_gtp",0):.4f}', flush=True)
    else:
        print(f'  {name}: ERROR - {data.get("message","")}', flush=True)

with open('/tmp/tier3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('ALL DONE', flush=True)
