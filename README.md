# LQCD Toolkit

**A convention-explicit lattice-QCD workflow for source generation, propagator solves, meson correlators, and correlated analysis.**

[中文介绍](README.zh-CN.md) · [User guide](docs/USER_GUIDE.md) · [Physics conventions](docs/PHYSICS_CONVENTIONS.md) · [Scientific validation](docs/SCIENTIFIC_VALIDATION.md)

Optional [scientific plotting](docs/plotting.md): install `pip install -e '.[plot]'`
for effective energy/mass, correlator and fixed-fit residuals, form factors and
supplied z-expansion bands. The Python/JSON interface exports PDF/SVG/PNG with
explicit uncertainty labels and hash-based provenance; analysis alone does not
require matplotlib.

LQCD Toolkit connects a physical configuration to a complete, traceable calculation: construct sources, solve propagators through an explicitly selected backend, construct and solve sequential sources, contract two- and three-point functions, write the newly generated correlators directly into SQLite, and optionally fit selected channels. Generated selection manifests carry the measurement identities needed for analysis; users do not need to assemble database file IDs by hand.

This repository contains only the new general-purpose implementation, tests, synthetic fixtures, and configuration templates. Historical source archives and the original repository history remain private and are not distributed here.

The repository also provides read-only access to compatible existing correlator databases. The shared reader supports both `lqcd.opaque-chunks.v1` and `v2`; [existing-data format profiles](docs/LEGACY_FORMATS.md) adapt explicit `cf` naming, Decimal mass labels and distinct time mappings without duplicating the numerical core. Numerical decoding, measurement selection, statistical resampling, and physical interpretation remain separate operations, so an intact file is never silently treated as a scientifically identified measurement.

**Current scientific status:** the small explicit-matrix workflow and numerical analysis have been tested, including an end-to-end mathematical example. The actual external XQCD action has **not** been validated numerically by this project. A successful configuration check or process invocation does not certify an action, a renormalization convention, or a physical form factor. External results retain the status `external-numerics-unverified`.

## What is included

| Area | Capabilities |
|---|---|
| Source generation | Point sources and reproducible spin/color-diluted Z3 spatial grids; explicit positions, geometry, and seeds |
| Solvers | Small dense-matrix reference solves with true residual checks; an external XQCD stdin/read-source adapter with explicit executable, action inputs, and residual contract |
| Sequential workflow | Spectator-based sequential sources, initial-active-mass sequential inversion, CPU reference C2/C3 contractions |
| Storage | Direct creation of a sealed SQLite store using `lqcd.opaque-chunks.v1`, original payload hashes, provenance, and automatic selection sidecars |
| Existing data | Read-only listing, inspection, integrity verification, and explicit-profile numeric decoding |
| Statistics | Within-configuration source averages, explicit configuration alignment, equal-size bins, jackknife, and paired bootstrap utilities |
| Fits | Correlated multi-state C2 fits and joint initial-C2/final-C2/C3 fits, covariance SVD cuts, analytic joint-model derivatives, parameter-rank and fit-failure checks |
| Physics kernels | Explicit-overlap spectral models, momentum and Minkowski q², restricted P→P vector-current extraction, real z mapping, and a simple pole-factored z series |

This is a CPU reference implementation, not a replacement for a production Dirac-action code. It does not automatically extract P→V or V→P form factors. A general gamma contraction alone does not establish support for a particular vector-meson decay.

## Install and start

Use Python 3.10 or newer, with dependency versions compatible with `pyproject.toml`. NumPy and SciPy provide the numerical runtime.

```sh
python -m pip install -e .
```

To try a complete mathematical example without a gauge field or external solver:

```sh
python tools/create_reference_example.py examples/my_reference
python run.py check examples/my_reference/physics.json
python run.py run examples/my_reference/physics.json
```

This creates three 48×48 reference matrices and runs source generation through SQLite output. It uses one configuration and disables statistical analysis; it is not a physical QCD dataset. The example and output directories must be new.

Copy `configs/physics.template.json` to `physics.json`, fill its physical inputs, and set `template` to `false`. The template is intentionally incomplete. For an external solve, copy and configure `profiles/xqcd.template.json` once for your installation, including its referenced solver-input template. Point the physical configuration at that backend profile.

```sh
python run.py check physics.json
python run.py run physics.json
```

`check` validates the planned measurements, paths, geometry, backend contract, and memory estimate without starting a solver. `run` creates a new output directory and retains intermediate products and provenance. Existing output directories are not overwritten; choose a new destination for another run.

A physical configuration describes:

- the ensemble and explicit configuration IDs;
- lattice geometry and source positions;
- quark-mass labels and their **ordered** initial-active, final-active, and spectator roles;
- initial/final interpolating operators, the current, integer momentum modes, and source separation;
- a backend profile, memory budget, and optional analysis settings.

For example, this is a **process fragment**, not a standalone runnable configuration:

```json
{
  "initial_quark": "c",
  "final_quark": "d",
  "spectator_quark": "s",
  "initial_operator": "g5",
  "final_operator": "g5",
  "current": "g4",
  "momentum_initial": [0, 0, 0],
  "momentum_final": [1, 0, 0],
  "separation": 24
}
```

These labels select configured inputs. They do not calibrate quark masses or prove a physical channel assignment. Use a point source for this nonzero-momentum example. The current Z3-grid estimator supports **zero initial and final momentum only**, with an explicit `grid_estimator` of `zero-momentum-source-average`.

For a mathematical check, select `explicit-matrix-reference` with `scientific_mode="mathematical-validation"` and provide all three operator matrices. This mode verifies linear algebra and data flow; its matrices are not silently substituted for an external lattice action. See the [user guide](docs/USER_GUIDE.md) for complete configurations and output layout.

## Inspect and fit an existing database

The following commands read a compatible sealed database. Fit results are written to a new result file, leaving the database unchanged. The example file ID is illustrative; obtain actual IDs with `list`.

```sh
python run.py list results/correlators/data.sqlite --limit 20
python run.py inspect results/correlators/data.sqlite --id 1
python run.py inspect results/correlators/data.sqlite --id 1 --profile savedata.pickle.numeric.v1
python run.py verify results/correlators/data.sqlite --all-payloads
python run.py fit2 fit2.json --output fit2-result.json --verify-database-hash
python run.py joint joint.json --output joint-result.json --verify-database-hash
```

`fit2.json` and `joint.json` are analysis settings, not physical-generation configurations. They reference explicit selection manifests and fixed fit windows. Source/configuration membership, time mapping, component projection, and covariance settings must be known before fitting. Newly generated databases supply selection manifests automatically.

The same operations are accessible from Python:

```python
from lqcd_analysis.database import CorrelatorStore
from lqcd_analysis.workflow import plan_workflow, run_workflow
from lqcd_analysis.analysis import run_fit_file
from lqcd_analysis.joint_analysis import run_joint_file

plan = plan_workflow("physics.json")  # validation only

with CorrelatorStore("results/correlators/data.sqlite") as store:
    for record in store.iter_files(limit=5):
        print(record.id, record.path, record.sha256)

# Explicit opt-in actions:
# run = run_workflow("physics.json")
# two_point = run_fit_file("fit2.json", verify_database_hash=True)
# joint = run_joint_file("joint.json", verify_database_hash=True)
```

## Conventions that affect results

The public generation profile is **`cpu-sequential-reference.v1`**. Its conventions are documented rather than inferred from array shapes:

- Raw propagators have big-endian float64 real/imaginary storage with source spin/color first. C2 masses are ordered `(spectator, active)`; C3 masses are `(initial, final, spectator)`.
- The sequential source is `Gamma_initial × spectator` at the chosen time slice, with a negative initial-momentum Fourier phase. The sequential inversion uses the initial active mass.
- Stored CPU sequential C3 index `j` maps to `(source_time + separation - j) % NT`. It runs from the initial insertion toward the final insertion, in decreasing raw lattice time.
- The reference tensor label uses `sigma_ij = gamma_i @ gamma_j` for distinct indices, **without an additional factor of i**.
- Complex C3 data are retained until an explicit phase and normalization projection is supplied. No automatic Euclidean-to-Minkowski conversion is applied.
- Z3-grid contractions are divided by their support-site count. They are stochastic estimators of a source average, not exact point-source averages in a single noise realization.

See the [English](docs/PHYSICS_CONVENTIONS.md) or [Chinese](docs/PHYSICS_CONVENTIONS.zh-CN.md) convention reference before combining generated and historical data.

## Limits and reproducibility

The dense reference backend is limited to dimension 4096, where the dimension is `12 * NT * NZ * NY * NX`. The tensor workflow also checks a conservative memory estimate against an explicit budget; the default budget is 1 GiB. A conventional production volume can require many GiB for even one propagator. Raising the budget does not make this implementation a distributed or GPU contraction engine.

The external executable and MPI environment are supplied by the user. Backend action fields, boundary conditions, dependencies, and residual interpretation are explicit. The adapter checks process completion, output size, and the supplied residual contract, but these checks cannot establish the correctness of that contract or executable.

Numerical decoders accept bounded numeric grammars. Unknown formats stay unsupported rather than falling back to executable pickle loading. The SQLite reader verifies each returned payload's size and SHA256; a full database hash check is separately available. Numerical reproducibility also requires retaining the action, masses, geometry, source definition, current convention, and statistical assumptions.

## Documentation and contributions

- [User guide](docs/USER_GUIDE.md): configuration, commands, outputs, and package modules.
- [Physics conventions](docs/PHYSICS_CONVENTIONS.md): axes, index contractions, spectra, form-factor definitions, and units.
- [Scientific validation](docs/SCIENTIFIC_VALIDATION.md): tested claims, resolved counterexamples, and remaining external validation work.
- [Contributing](CONTRIBUTING.md): reproducible changes, tests, documentation, and public-data hygiene.
- [Contributors](CONTRIBUTORS.md): contributor attribution, including AI assistance.

`diagnostics.py` and `legacy_profiles.py` are new implementations. The former provides effective-energy diagnostics; the latter adapts supported existing-data naming and time conventions through the single shared database reader, decoder, and analysis core. Format compatibility does not include or execute historical scripts. No old source archive, original repository history, production dataset, or private deployment configuration is included in this repository. See [release validation](docs/VALIDATION_RESULTS.md) and [dependency/license boundaries](THIRD_PARTY.md).

The public repository is released under the [MIT License](LICENSE). Third-party dependencies remain subject to their own licenses. Code availability and an open-source license do not certify scientific results.
