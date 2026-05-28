"""Tier 4: Null controls — STAT3 (deliberate null) vs SHP2 (known allosteric)."""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATASET_CONFIG_FILE', 'tokyoeyes_dataset_config.json')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_science.sub_agents.dtie.tools import run_dtie_pipeline

# Tier 4: Null controls
# STAT3: 1BG1 (SH2 domain) — no known allosteric mechanism, should show
# fewer doorways and lower uncertainty-gated landmarks than SHP2
# Use same structure for both GDP/GTP slots (self-comparison = null signal)
# vs SHP2 which has genuine conformational change (2SHP vs 6MCF)

pairs = [
    # STAT3 self-comparison (null — same structure both slots)
    ('1BG1', '1BG1', 'STAT3_null_self'),
    # SHP2 genuine comparison (positive control — known allosteric)
    ('2SHP', '6MCF', 'SHP2_positive_control'),
    # Additional null: Ubiquitin (1UBQ) — small, rigid, no allostery
    ('1UBQ', '1UBQ', 'Ubiquitin_null_self'),
]

results = {}
for gdp, gtp, name in pairs:
    print(f'Running {name}: {gdp} vs {gtp}...', flush=True)
    result = run_dtie_pipeline(
        gdp_pdb_id=gdp, gtp_pdb_id=gtp,
        pipeline_mode='source_leak_v4',
        n_landmarks=200, effector_sites='',
        checkpoint='pipeline',
    )
    data = json.loads(result)
    results[name] = data
    status = data['status']
    if status == 'ok':
        p2 = data.get('phase2', {})
        p4b = data.get('phase4b', {})
        print(f'  {name}: {p2.get("n_doorways")} doorways, {p2.get("n_constitutive")} constitutive', flush=True)
        print(f'    L2 GDP={p4b.get("lambda_2_gdp",0):.4f} GTP={p4b.get("lambda_2_gtp",0):.4f}', flush=True)
    else:
        print(f'  {name}: ERROR - {data.get("message","")}', flush=True)

with open('/tmp/tier4_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('ALL DONE', flush=True)
