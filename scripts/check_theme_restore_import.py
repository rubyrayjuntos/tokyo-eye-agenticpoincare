from experiments.training.v8.equ_theme_restore import GATE_ID, DEFAULT_SPINE_INIT
from experiments.training.v8 import run_tokyo_eye_equ_theme_restore as r
print("OK", GATE_ID, DEFAULT_SPINE_INIT.is_file(), r.TAG)
