# Ledger B — Phase 4b Hub–Literature Map

**Recorded:** 2026-07-20T20:38:29.337368+00:00  
**Checkpoint:** `/app/checkpoints/v66/runs/fix1_s4_sparsity_confirm_continue_v2/v66_sparsity_champion.pt`  
**Scope:** Interpretative only — does not change Ledger B Pass form or pre-reg *I*.  
**Policy (2026-07-21):** Option 1 closeout — interface recall **FAIL retained**; this map is the appended discovery explanation. See [`ledger-b-interface-policy-closeout.md`](ledger-b-interface-policy-closeout.md).

## Method

1. Forward-knockout `out_effect` on the sparsity champion (same definition as Ledger B).
2. *H* = top 10% residues by `out_effect`.
3. Partition into **H∩I** (overlap with locked pre-reg interface) and **H\\I** (discovery candidates).
4. Annotate with domain buckets + curated motif notes. Machine JSON: `checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_map.json`.
5. Grade artifact patched with `hub_resseqs` / `hubs_ranked` / `hubs_annotated`.

## Executive read

| Structure | \|H\| | \|H∩I\| | \|H\\I\| | Domain skew of *H* |
|-----------|------|--------|--------|---------------------|
| 2SHP | 50 | 2 | 48 | PTP=23, N-SH2=20, C-SH2=7 |
| 3PP0 | 29 | 4 | 25 | C-lobe C-terminal=12, C-lobe (pre-DFG)=7, N-lobe=7, activation segment / A-loop=3 |

**Takeaway:** Knockout hubs are **not** concentrated on the locked latch/tunnel (SHP2) or ATP-spine (SRC) sets. They skew to **PTP + N-SH2 bulk** (2SHP) and **C-lobe / αC-flank packing** (3PP0), with only thin bleed into pre-reg *I*.

## 2SHP (SHP2)

- *n* residues: 491; |*H*|=50; |H∩I|=2; |H\\I|=48
- recall@top-10% (recomputed): **0.074** (bar 0.25)
- literature→auth offset: 0

### H∩I (partial canonical hits)

| rank | auth | lit | aa | domain | note | out_effect |
|-----:|-----:|----:|:--:|:-------|:-----|----------:|
| 11 | 102 | 102 | LEU | N-SH2 | N-SH2–PTP tunnel-2 / interface band | 0.1141 |
| 27 | 100 | 100 | TYR | N-SH2 | N-SH2–PTP tunnel-2 / interface band | 0.1056 |

### H\\I (discovery candidates)

| rank | auth | lit | aa | domain | note | out_effect |
|-----:|-----:|----:|:--:|:-------|:-----|----------:|
| 1 | 310 | 310 | ILE | PTP | PTP interior — near WPD-loop / catalytic neighborhood (not tunnel wall) | 0.1239 |
| 2 | 290 | 290 | VAL | PTP | PTP hydrophobic core / β-sheet packing | 0.1236 |
| 3 | 308 | 308 | ASN | PTP | PTP — adjacent to 310 cluster | 0.1222 |
| 4 | 457 | 457 | VAL | PTP | PTP — immediate neighbor of catalytic Cys459 (secondary set); active-site adjacency without being in primary I | 0.1205 |
| 5 | 354 | 354 | VAL | PTP | PTP mid-domain | 0.1199 |
| 6 | 352 | 352 | VAL | PTP | PTP mid-domain | 0.1186 |
| 7 | 32 | 32 | ARG | N-SH2 | Classic N-SH2 pTyr-binding pocket residue (Arg32) — high-utility recognition, not in latch/tunnel I | 0.1178 |
| 8 | 386 | 386 | ARG | PTP | PTP — Arg in catalytic lobe (distinct from HRD numbering in kinases) | 0.1161 |
| 9 | 467 | 467 | GLY | PTP | no curated interface annotation — candidate discovery hub | 0.1153 |
| 10 | 418 | 418 | TYR | PTP | no curated interface annotation — candidate discovery hub | 0.1146 |
| 12 | 137 | 137 | VAL | C-SH2 | no curated interface annotation — candidate discovery hub | 0.1139 |
| 13 | 343 | 343 | ARG | PTP | no curated interface annotation — candidate discovery hub | 0.1120 |
| 14 | 45 | 45 | VAL | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1114 |
| 15 | 28 | 28 | SER | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1106 |
| 16 | 387 | 387 | ASN | PTP | no curated interface annotation — candidate discovery hub | 0.1103 |
| 17 | 329 | 329 | ALA | PTP | no curated interface annotation — candidate discovery hub | 0.1103 |
| 18 | 112 | 112 | TRP | C-SH2 | no curated interface annotation — candidate discovery hub | 0.1095 |
| 19 | 99 | 99 | LYS | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1094 |
| 20 | 8 | 8 | HIS | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1084 |
| 21 | 504 | 504 | MET | PTP | no curated interface annotation — candidate discovery hub | 0.1079 |
| 22 | 172 | 172 | ILE | C-SH2 | no curated interface annotation — candidate discovery hub | 0.1071 |
| 23 | 458 | 458 | HIS | PTP | no curated interface annotation — candidate discovery hub | 0.1070 |
| 24 | 473 | 473 | ASP | PTP | no curated interface annotation — candidate discovery hub | 0.1065 |
| 25 | 75 | 75 | ALA | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1060 |
| 26 | 98 | 98 | LEU | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1056 |
| 28 | 46 | 46 | ARG | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1054 |
| 29 | 55 | 55 | LYS | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1052 |
| 30 | 31 | 31 | ALA | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1048 |
| 31 | 9 | 9 | PRO | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1037 |
| 32 | 404 | 404 | SER | PTP | no curated interface annotation — candidate discovery hub | 0.1037 |
| 33 | 501 | 501 | ARG | PTP | no curated interface annotation — candidate discovery hub | 0.1020 |
| 34 | 73 | 73 | THR | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1017 |
| 35 | 305 | 305 | ILE | PTP | no curated interface annotation — candidate discovery hub | 0.1014 |
| 36 | 505 | 505 | VAL | PTP | no curated interface annotation — candidate discovery hub | 0.1013 |
| 37 | 17 | 17 | GLU | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1005 |
| 38 | 74 | 74 | LEU | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1004 |
| 39 | 147 | 147 | PHE | C-SH2 | no curated interface annotation — candidate discovery hub | 0.1003 |
| 40 | 19 | 19 | LEU | N-SH2 | no curated interface annotation — candidate discovery hub | 0.1002 |
| 41 | 211 | 211 | GLN | C-SH2 | no curated interface annotation — candidate discovery hub | 0.1000 |
| 42 | 309 | 309 | ILE | PTP | no curated interface annotation — candidate discovery hub | 0.0999 |
| 43 | 69 | 69 | GLU | N-SH2 | no curated interface annotation — candidate discovery hub | 0.0990 |
| 44 | 202 | 202 | MET | C-SH2 | no curated interface annotation — candidate discovery hub | 0.0989 |
| 45 | 215 | 215 | PRO | C-SH2 | no curated interface annotation — candidate discovery hub | 0.0986 |
| 46 | 439 | 439 | LEU | PTP | no curated interface annotation — candidate discovery hub | 0.0983 |
| 47 | 42 | 42 | THR | N-SH2 | no curated interface annotation — candidate discovery hub | 0.0983 |
| 48 | 71 | 71 | PHE | N-SH2 | no curated interface annotation — candidate discovery hub | 0.0982 |
| 49 | 466 | 466 | THR | PTP | no curated interface annotation — candidate discovery hub | 0.0980 |
| 50 | 344 | 344 | MET | PTP | no curated interface annotation — candidate discovery hub | 0.0980 |

## 3PP0 (SRC)

- *n* residues: 286; |*H*|=29; |H∩I|=4; |H\\I|=25
- recall@top-10% (recomputed): **0.078** (bar 0.25)
- literature→auth offset: 433

### H∩I (partial canonical hits)

| rank | auth | lit | aa | domain | note | out_effect |
|-----:|-----:|----:|:--:|:-------|:-----|----------:|
| 7 | 862 | 429 | THR | activation segment / A-loop | Activation-loop window (pre-reg S2) | 0.2147 |
| 10 | 850 | 417 | ASN | activation segment / A-loop | Activation-loop window (pre-reg S2) | 0.2092 |
| 11 | 848 | 415 | ALA | activation segment / A-loop | Activation-loop window (pre-reg S2) | 0.2073 |
| 18 | 819 | 386 | SER | C-lobe (pre-DFG) | HRD catalytic loop | 0.1945 |

### H\\I (discovery candidates)

| rank | auth | lit | aa | domain | note | out_effect |
|-----:|-----:|----:|:--:|:-------|:-----|----------:|
| 1 | 800 | 367 | LEU | C-lobe (pre-DFG) | C-lobe hydrophobic packing (αE/αF vicinity) — not hinge/DFG/A-loop primary motifs | 0.2369 |
| 2 | 798 | 365 | THR | C-lobe (pre-DFG) | C-lobe packing near 367 cluster | 0.2367 |
| 3 | 738 | 305 | ILE | N-lobe | N-lobe — αC-helix neighborhood (αC Glu is lit 310; this is packing/flank) | 0.2348 |
| 4 | 801 | 368 | MET | C-lobe (pre-DFG) | C-lobe packing cluster with 365/367 | 0.2313 |
| 5 | 947 | 514 | CYS | C-lobe C-terminal | Far C-lobe / C-terminal KD — distal to ATP site | 0.2230 |
| 6 | 747 | 314 | LYS | N-lobe | N-lobe — αC flank (near Glu310 salt-bridge partner Lys295 axis) | 0.2220 |
| 8 | 751 | 318 | ALA | N-lobe | N-lobe — αC / β4 region packing | 0.2124 |
| 9 | 987 | 554 | VAL | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.2097 |
| 12 | 891 | 458 | LEU | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.2060 |
| 13 | 968 | 535 | ARG | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.2030 |
| 14 | 886 | 453 | ILE | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.2022 |
| 15 | 893 | 460 | SER | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.2019 |
| 16 | 749 | 316 | PRO | N-lobe | no curated KD-motif annotation — candidate discovery hub | 0.1982 |
| 17 | 989 | 556 | ILE | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.1972 |
| 19 | 805 | 372 | CYS | C-lobe (pre-DFG) | no curated KD-motif annotation — candidate discovery hub | 0.1927 |
| 20 | 826 | 393 | CYS | C-lobe (pre-DFG) | no curated KD-motif annotation — candidate discovery hub | 0.1923 |
| 21 | 915 | 482 | LEU | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.1923 |
| 22 | 737 | 304 | GLY | N-lobe | no curated KD-motif annotation — candidate discovery hub | 0.1900 |
| 23 | 735 | 302 | TYR | N-lobe | no curated KD-motif annotation — candidate discovery hub | 0.1885 |
| 24 | 928 | 495 | ALA | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.1881 |
| 25 | 988 | 555 | VAL | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.1879 |
| 26 | 796 | 363 | LEU | C-lobe (pre-DFG) | no curated KD-motif annotation — candidate discovery hub | 0.1876 |
| 27 | 946 | 513 | ILE | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.1874 |
| 28 | 887 | 454 | LYS | C-lobe C-terminal | no curated KD-motif annotation — candidate discovery hub | 0.1869 |
| 29 | 734 | 301 | VAL | N-lobe | no curated KD-motif annotation — candidate discovery hub | 0.1864 |

## Structure-specific interpretation

### 2SHP (SHP2)

- **H∩I** is only **Y100** and **L102** (tunnel-2 band) — ranks mid-pack, not the top knockout sites. Thin latch/tunnel concordance.
- **Top hubs** sit in the **PTP domain** (I310, V290, N308, V457, …) and **N-SH2** (notably **R32**, a canonical pTyr-pocket residue that was *not* in latch/tunnel *I*).
- **V457** is adjacent to catalytic **C459** (secondary-only in pre-reg): the model weights catalytic-neighborhood geometry more than the SHP099 tunnel wall.
- Discovery hypothesis to test next: hubs track **autoinhibited N-SH2↔PTP closure mechanics / catalytic lobe packing**, not the drug-tunnel annotation set.

### 3PP0 (SRC KD)

- **H∩I** = lit **S386** (HRD) + A-loop **A415 / N417 / T429** — activation-segment bleed, not P-loop / Lys295 / DFG dominance.
- **Top hubs** are **C-lobe packing (lit ~365–368)** and **αC flank (lit ~305 / 314 / 318)**, plus distal C-terminal KD sites — consistent with **lobe-coupling / αC dynamics** rather than ATP-site residue lists.
- Discovery hypothesis: flow hubs emphasize **N↔C lobe mechanical coupling**, which can be orthogonal to shortest-path betweenness and to static ATP-site annotations.

## Reading rules (lockdown)

- Do **not** lower the 0.25 recall bar from this map.
- Do **not** promote H\\I residues into *I* without a new pre-registration stamp.
- Next experimental step (Phase 4c): OOD / activated-state migration — ask whether flow moves onto cryptic or open-state interfaces when the resting autoinhibited geometry changes.

