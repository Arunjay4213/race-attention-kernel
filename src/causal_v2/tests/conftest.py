"""Makes src/causal_v2 (reference, build) and src/ (race_baseline) importable from the tests."""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # src/causal_v2
sys.path.insert(1, str(_HERE.parents[2]))  # src
