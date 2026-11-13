# LQCD Toolkit user guide

## What the supported code does

The package separates physical definitions, numeric calculation, data provenance
and machine setup. A user configuration describes an ensemble, quark masses,
operators, momenta, source positions and source-to-sink separation. A reusable
backend profile describes the installed inverter and its verified input/output
contract. Neither layer contains a default machine address or account.

| Module | Responsibility |
| --- | --- |
| `generation.py` | Point/Z3 sources, checked propagator solve, sequential sources and CPU-reference C2/C3 contractions |
| `workflow.py` | One physical configuration, preflight of every task, sequential execution and optional analysis |
| `writer.py` | Direct numeric correlator output to SQLite; byte roundtrip and scientific selection validation before sealing |
| `database.py` | Read-only versioned store access, provenance checks and ordered chunk length/SHA256 verification |
| `decoders.py` | Explicit numeric pickle/NPY/NPZ/text profiles with bounded decoding; no serialized callable execution |
| `selection.py` | Literal measurement identity, mass/operator/momentum checks and explicit time transformations |
| `legacy_profiles.py` | New CF-format compatibility adapters that reuse the same reader, decoder, selection, and analysis core |
| `statistics.py` | Configuration IDs, source averages, common sample alignment, bins, jackknife and paired bootstrap |
| `fitting.py` | Correlated least squares, covariance-mode cutoff, Jacobian rank, fit diagnostics and complete jackknife fits |
| `analysis.py` | SQLite-to-periodic-C2 fit |
| `joint_analysis.py` | Joint initial C2, final C2 and C3 spectral fit with an analytic Jacobian |
| `diagnostics.py` | Log/cosh effective energies and complete jackknife propagation |
| `physics.py` | Explicit spectral kernels, momentum/q² algebra, restricted P→P extraction and an explicit pole-factored z series |

The shared reader accepts `lqcd.opaque-chunks.v1` and `v2`; the generated writer produces v1. New existing-data writer profiles extend that same reader and selection layer. They contain format compatibility code, not historical scripts, and do not introduce a second solver, resampler, or fitting framework.
See [the storage contract](STORE_FORMAT.md) and [physical conventions](PHYSICS_CONVENTIONS.md).

## Install and try the complete small example

Use Python 3.10 or later with NumPy and SciPy. From the repository:

```bash
python -m pip install -e .
python tools/create_reference_example.py examples/my_reference
python run.py check examples/my_reference/physics.json
python run.py run examples/my_reference/physics.json
```

The tool creates three explicit small matrices and a physical-format
configuration. The run actually creates the source, solves four linear systems
with twelve spin/color right-hand sides each, builds a sequential source,
contracts three correlators and seals a SQLite database. It does **not** use a
physical gauge field or implement a production fermion action. The example's
`scientific_mode` is `mathematical-validation` and analysis is disabled because
one configuration is insufficient to estimate statistical errors.

The output contains `products/`, `correlators/data.sqlite`, per-channel
`correlators/selections/` and `workflow.json`. Output directories must be new;
rerunning into an existing destination fails without deleting it.

## Fill in a physical calculation

Copy `configs/physics.template.json` to a new local configuration. Complete it
and set `template` to `false`. Relative input paths are resolved against that
configuration's directory; paths in a separate backend profile are resolved
against the profile directory.

| Field | Meaning |
| --- | --- |
| `ensemble` | Stable identity of the gauge ensemble; never inferred from filenames |
| `geometry` | Integer `nt,nz,ny,nx` extents |
| `quark_masses` | Flavor labels mapped to positive decimal strings in lattice units |
| `process.*_quark` | Explicit initial active, final active and spectator flavor routing |
| `initial_operator`, `final_operator`, `current` | Literal gamma labels in the documented Euclidean basis |
| `momentum_initial`, `momentum_final` | Integer Fourier mode triples `[nx,ny,nz]` |
| `separation` | `0 < T < NT` |
| `sources.positions` | `[x,y,z,t]` coordinates, at most one source measurement per configuration/time in this version |
| `configurations` | Unique canonical decimal configuration IDs and their gauge files |
| `backend` | An inline backend object or path to a reusable backend JSON |
| `max_memory_bytes` | Explicit allowed memory for the reference calculation |
| `analysis` | `null`, a two-point fit request, or a joint C2/C3 request |

Check before running:

```bash
python run.py check my_physics.json
python run.py run my_physics.json
```

`check` starts no solver and creates no output. It validates input files,
geometry, mass routing, source and momentum restrictions, every inverter input
template, the reference memory estimate and requested analysis settings. It
cannot establish physical correctness of a supplied action, source estimator,
renormalization constant or fit window.

Point sources support the declared nonzero momentum projection. Z3 grids require
`base_seed`, grid spacings dividing the spatial extents and
`grid_estimator="zero-momentum-source-average"`; both external momenta must be
zero in this reference profile. Results are divided by the grid support count.
The same grid sites are not independent gauge configurations.

## Connect an installed production inverter once

Complete `profiles/xqcd.template.json` and `generation/xqcd_input.template`.
The profile supplies an explicit executable/launcher argument list, action
evidence, dependency files, residual log grammar and threshold. The input
template supplies all action and solver settings; no production values are
guessed. The adapter uses standard input, validates raw big-endian float64
output, records input/output/dependency hashes and requires complete residual
evidence. It never submits scheduler jobs or logs into a machine automatically.

Run on an allocated compute resource. Choose local/MPI/scheduler execution in
your own runtime configuration. Do not combine multiple independent launchers by
accident. The CPU reference implementation keeps full tensors: its default
1 GiB estimate rejects large volumes rather than silently allocating them.
There is no claim of production streaming or GPU performance.

An external run retains `backend_status="external-numerics-unverified"` until
the operator has independently established binary/source correspondence,
`read_source` behavior, contact-term/normalization conventions and residual
meaning. Successful process exit and a matching file size do not establish
these scientific properties.

## Analyze stored correlators

Generated correlators receive selection manifests automatically. For historical
stores, explicit writer and measurement metadata must be supplied. Raw bytes
can always be inspected without inventing a physical channel.

```bash
python run.py list run-output/correlators/data.sqlite --limit 10
python run.py inspect run-output/correlators/data.sqlite --id 1 --profile savedata.pickle.numeric.v1
python run.py verify run-output/correlators/data.sqlite --all-payloads
python run.py fit2 my_fit.json --output c2-result.json
python run.py joint my_joint_fit.json --output joint-result.json
```

The `fit2` settings use version `lqcd.fit2.v1`, database/selection paths, an exact
configuration order, bin size, fixed fit-time list, covariance cutoff and an
explicit periodic positive diagonal model. State energies are increasing via
positive gaps; overlaps are positive magnitudes. Both endpoints must use the
same literal operator and the real projection must be physically justified.
The model's bounds and starting energies/overlaps are explicit.

`generation/joint_fit.template.json` documents joint settings. Supply separate
initial/final C2 and complex C3 selections, the C3 physical time mapping,
current phase/scale evidence, fixed windows, model sectors and a real matrix
`M[final_state,initial_state]` with bounds. C3 contact endpoints are excluded.
An analytic Jacobian prevents finite-difference noise from turning an exact
spectral degeneracy into apparent identifiability. A single separation can
still leave multi-state transition matrix elements unidentifiable; such fits
are rejected rather than assigned finite errors.

Each configuration is equally weighted after averaging its source measurements.
Channels are aligned by IDs, not array positions. Bins follow the supplied chain
order and never drop a remainder. All jackknife fits share one chosen window
and full-sample covariance; any failed replicate terminates the calculation.
Independent ensembles are not pooled by giving unrelated configurations the
same row index. P-values are asymptotic diagnostics, not calibrated coverage
claims after model/window selection.

## Diagnostics and derived physics

`diagnostics.effective_energy` implements `log(C[t]/C[t+1])` or the local cosh
relation with explicit spacing and time origin. It requires finite positive
real input and a valid real domain. The jackknife variant retains every
replicate and reports the failed deleted-configuration identity; it does not
clip, take absolute values or drop failed samples.

`physics.pp_form_factors` accepts an already renormalized, real contravariant
Minkowski current and continuum on-shell kinematics. It rejects rank-deficient
zero recoil and avoids division by q², preserving `f0(0)=f+(0)`. It is algebraic
extraction, not a noisy-current correlated fit or an automatic Euclidean-phase
conversion. P→V and V→P form-factor extraction is not implemented automatically.
The z-series helper is an explicit pole-factored polynomial, not a default
BCL/BGL model. See the convention and validation documents before using these
tools for physical inference.

## Repository scope and contribution

This repository contains new general-purpose implementation code, tests, synthetic fixtures, and templates. `diagnostics.py` and `legacy_profiles.py` are new modules; supporting an existing data format does not bundle or execute its historical writer. Old source archives and original repository history remain private. Third-party/vendor source and real scientific data are not bundled. See [license boundaries](../THIRD_PARTY.md),
[contributors](../CONTRIBUTORS.md) and [contribution rules](../CONTRIBUTING.md).
