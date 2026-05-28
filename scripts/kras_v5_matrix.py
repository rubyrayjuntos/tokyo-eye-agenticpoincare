"""KRAS v5 full comparison matrix: WT/G12D × active/inactive."""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATASET_CONFIG_FILE', 'tokyoeyes_dataset_config.json')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_science.sub_agents.dtie.tools import run_dtie_pipeline

# KRAS structures:
# WT active (GTP): 6GOD
# WT inactive (GDP): 4OBE
# G12D active (GTP): 6GOF
# G12D inactive (GDP): 4DSO

pairs = [
    ('6GOD', '6GOF', 'WT_active_vs_G12D_active'),
    ('6GOD', '4OBE', 'WT_active_vs_WT_inactive'),
    ('6GOD', '4DSO', 'WT_active_vs_G12D_inactive'),
    ('6GOF', '4DSO', 'G12D_active_vs_G12D_inactive'),
]

results = {}
for pdb_a, pdb_b, name in pairs:
    print(f'Running {name}: {pdb_a} vs {pdb_b}...', flush=True)
    result = run_dtie_pipeline(
        gdp_pdb_id=pdb_a, gtp_pdb_id=pdb_b,
        pipeline_mode='source_leak_v4',
        n_landmarks=200, effector_sites='',
        checkpoint='v5',
    )
    data = json.loads(result)
    results[name] = data
    if data['status'] == 'ok':
        p2 = data.get('phase2', {})
        p4b = data.get('phase4b', {})
        print(f'  {name}: {p2.get("n_doorways")} doorways, {p2.get("n_constitutive")} const', flush=True)
        print(f'    L2: {p4b.get("lambda_2_gdp",0):.4f} / {p4b.get("lambda_2_gtp",0):.4f}', flush=True)
        print(f'    Hubs: {p4b.get("top_gdp_hub")} / {p4b.get("top_gtp_hub")}', flush=True)
    else:
        print(f'  {name}: ERROR - {data.get("message","")}', flush=True)

with open('/tmp/kras_v5_matrix.json', 'w') as f:
    json.dump(results, f, indent=2)
print('ALL DONE', flush=True)
