"""User-facing entry point. Example: python run.py check configs/physics.json."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# A small reference run should not acquire every CPU by default. Production MPI
# allocation is explicit in the user's backend configuration.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from lqcd_analysis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
