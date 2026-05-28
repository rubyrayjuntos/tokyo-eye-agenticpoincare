"""v5 Full Validation: Tiers 2-4 in one run."""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DATASET_CONFIG_FILE', 'tokyoeyes_dataset_config.json')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_science.sub_agents.dtie.tools import run_dtie_pipeline

all_results = {}

# ── TIER 2: Pre-registered pathway proteins ──
tier2_pairs = [
    ('2SHP', '6MCF', 'T2_SHP2'),
    ('3EQI', '3PP1', 'T2_MEK1'),
    ('1ERK', '2ERK', 'T2_ERK2'),
]

# ── TIER 3: Matched-pair differential ──
tier3_pairs = [
    ('4OBE', '4LDJ', 'T3_KRAS_WT_vs_G12C'),
    ('4DSO', '4TQ9', 'T3_KRAS_G12D_vs_G12V'),
    ('3TV4', '4MNE', 'T3_BRAF_WT_vs_V600E'),
    ('1M17', '2ITV', 'T3_EGFR_WT_vs_L858R'),
]

# ── TIER 4: Null controls ──
tier4_pairs = [
    ('1BG1', '1BG1', 'T4_STAT3_null'),
    ('1UBQ', '1UBQ', 'T4_Ubiquitin_null'),
    ('2SHP', '6MCF', 'T4_SHP2_positive'),
]

all_pairs = tier2_pairs + tier3_pairs + tier4_pairs

for pdb_a, pdb_b, name in all_pairs:
    print(f'Running {name}: {pdb_a} vs {pdb_b}...', flush=True)
    result = run_dtie_pipeline(
        gdp_pdb_id=pdb_a, gtp_pdb_id=pdb_b,
        pipeline_mode='source_leak_v4',
        n_landmarks=200, effector_sites='',
        checkpoint='v5',
    )
    data = json.loads(result)
    all_results[name] = data
    if data['status'] == 'ok':
        p2 = data.get('phase2', {})
        p4b = data.get('phase4b', {})
        ids = p2.get('doorway_ids', [])
        resnums = sorted(set(int(m.group(0)) for x in ids for m in [re.search(r'\d+', x)] if m))
        print(f'  {p2.get("n_doorways")} doorways, {p2.get("n_constitutive")} const | '
              f'L2={p4b.get("lambda_2_gdp",0):.4f}/{p4b.get("lambda_2_gtp",0):.4f} | '
              f'hubs={p4b.get("top_gdp_hub")}/{p4b.get("top_gtp_hub")} | '
              f'res={resnums[:8]}', flush=True)
    else:
        print(f'  ERROR: {data.get("message","")}', flush=True)

with open('/tmp/v5_full_validation.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nALL DONE', flush=True)
