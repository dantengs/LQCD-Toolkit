# Scientific validation and remaining work

[English overview](../README.md) · [中文介绍](../README.zh-CN.md) · [Conventions](PHYSICS_CONVENTIONS.md)

This repository distinguishes software verification, mathematical validation, and validation of an actual lattice calculation. These are different claims. The public CPU profile is `cpu-sequential-reference.v1`.

## What has been checked

| Layer | Evidence and scope | What it does not establish |
|---|---|---|
| Raw tensor layout | Small complex tensors round-trip through the explicit big-endian float64 real/imaginary layout | Compatibility with every external producer |
| Gamma and contractions | Clifford algebra, gamma5 relations, ordered tensor convention, explicit spin sums, source/sink index routing | Correct external action, current renormalization, or identification of a physical state |
| Source construction | Point-source dilution, deterministic Z3 phases, sequential slice modulo, nonzero-momentum phase tests, zero-momentum grid normalization tests | Reproduction of another generator's RNG, gauge-field ensemble, or noise reduction scheme |
| Reference solve | Explicit dense matrices are solved with checked per-right-hand-side true relative residuals | A production lattice Dirac action |
| External adapter | A real subprocess test double exercises stdin, file output, expected residual records, and rejection paths | Numerical correctness of the real XQCD executable |
| Complete data flow | A small mathematical example connects source, primary and sequential solves, contractions, direct SQLite storage, and generated sidecars | A physical decay calculation or a benchmark against published form factors |
| Storage and decoding | Numeric dtype, endian/order, hashes, bounded grammars, malformed inputs, and generated-store readback | Measurement identity or scientific conventions inferred from bytes alone |
| Statistics | Source averages, ID alignment, bin membership, joint covariance, paired resampling, and failure handling | Configuration independence or an automatically adequate bin length |
| Fits | Synthetic parameter recovery, explicit time conversion, analytic derivative checks, rank-deficiency regressions, and failure detection | Absence of excited-state, thermal, discretization, finite-volume, or model-selection systematics |
| Physics helpers | Spectral normalization, complex-overlap conjugation, unit changes, moving-frame P→P recovery, q²=0 constraint, z and pole domains | Automatic conversion of unrenormalized Euclidean correlators into physical form factors |

Independent mathematical and numerical review complemented the implementation tests. This public repository contains the new general-purpose implementation, tests, synthetic fixtures, and templates. Its validation arguments and regression examples are stated here without requiring historical source archives, private paths, or access to private data. The data-format compatibility adapter is new code using the shared v1/v2 reader and numerical core.

## Important counterexamples retained as regression requirements

### A declared ZIP size is not an actual decompression bound

The decoder review constructed a small archive whose compressed stream produced more bytes than its declared uncompressed member size. A general ZIP reader could truncate the extra output and return a valid NPY prefix. That behavior violated the intended resource and exact-length contract.

The decoder now uses explicit bounded inflation with actual length, stream-end, trailing-data, and CRC checks. The regression uses a small in-memory fixture; it does not execute serialized code or require a large decompression allocation. This is a specific tested boundary, not a proof that arbitrary hostile inputs are harmless in every runtime.

### Coincident transition exponents are not independent information

For one separation T, the forward C3 matrix-element coefficient is

$$
K_{mn}(\tau)=\frac{Z_{f,m}Z_{i,n}}{4E_{f,m}E_{i,n}}
e^{-E_{f,m}T}e^{-(E_{i,n}-E_{f,m})\tau}.
$$

If $E_i=(0.3,0.7)$ and $E_f=(0.2,0.6)$, then K00 and K11 have the same time dependence. Only one combination of M00 and M11 is identifiable at fixed energies. Numerical finite-difference noise previously allowed a very small rank cutoff to report twelve independent parameters in this twelve-parameter example.

The joint model now supplies analytic derivatives in log-gap, log-overlap, and matrix-element coordinates to the optimizer and rank check. The original example reports **rank 11 of 12**, with no ordinary parameter covariance or p-value, and cannot pass the joint full-rank requirement. An independent central-difference comparison checks the derivative formulas away from the degenerate case. Near-degenerate fits still require stability studies; analytic differentiation does not create information absent from the data.

### A checked plan must carry the estimator it validated

The easy-workflow review found that a valid Z3 estimator choice could be checked in the physical configuration but omitted from the task passed to generation. The wrapper now validates and forwards the estimator, including its zero-momentum limitation. This prevents a checked Z3 plan from failing solely because that field was dropped.

## Reproducing software checks

From an installed editable checkout:

```sh
python -m pip install -e .
python -B -m unittest discover -s tests -v
```

For a focused check of the generation and analysis boundary:

```sh
python -B -m unittest discover -s tests -p test_generation.py -v
python -B -m unittest discover -s tests -p test_joint_analysis.py -v
python -B -m unittest discover -s tests -p test_easy_workflow.py -v
```

Tests use synthetic numeric fixtures and a process-contract test double where required. Persistent fixture directories may be retained for inspection. Check the current test output rather than treating a historical test count as a guarantee for a later revision. Historical operational scripts are not part of this repository or its supported test process.

## What remains unverified for a real external calculation

`external-numerics-unverified` means exactly that. Before relying on a real external action, a validation record should establish:

1. **Executable and action identity.** Record the actual binary and dependencies, action parameters, boundary conditions, precision, gauge input, and any source/output transformations. An action name in a JSON file is not sufficient.
2. **Source-to-output agreement.** Confirm that the supplied raw source layout is read correctly and that the returned object has the intended propagator definition. In particular, establish any overlap-action normalization or contact-term treatment rather than assuming a plain inverse.
3. **Residual meaning.** Tie parsed residual records to the actual source, mass, and right-hand sides. Specify whether they are true or recursive residuals, their norm/normalization, and any action-specific reconstruction effects.
4. **Controlled contraction comparison.** Compare the complete primary-plus-sequential path with an independently derived reference on a manageable case, including all Fourier phases, operator ordering, temporal boundaries, and normalization.
5. **Measurement and statistical provenance.** Identify independent gauge configurations, source and noise membership, chain order, missing measurements, and the binning/fit-window study.
6. **Physical interpretation.** Establish mass tuning, interpolator overlap conventions, current renormalization, polarization projections when needed, Euclidean-to-Minkowski mapping, and the remaining physical systematics.

These are the boundaries of the claims made here, not additional automatic actions performed by `check` or `run`. A successful adapter invocation checks the documented software contract; it does not complete this scientific validation record.

## Interpreting output status

| Status or field | Meaning |
|---|---|
| `explicit-operator-reference` | The selected solve uses supplied finite matrices for mathematical validation |
| `external-numerics-unverified` | An external action is involved; production numerical equivalence remains unverified |
| `physics_status="unconfirmed"` | Metadata do not certify the physical channel or phase convention |
| `validated-numerics` | Numerical fit checks passed under the supplied model, projection, covariance, and selections |
| `requires-bound-review` | At least one fit reached a parameter bound; ordinary unconstrained inference needs review |

Neither `validated-numerics` nor a database integrity seal upgrades `physics_status` or proves a physical conclusion. P-values are asymptotic diagnostics with caveats about estimated covariance, fit-window choice, constraints, and identifiability.

## 中文说明

本项目已经验证的是软件接口、小规模线性代数、显式指标收缩、数据流和合成数据统计分析。完整数学示例通过，不等于真实外部 XQCD action、重整化或物理形状因子已经验证。`external-numerics-unverified` 必须保留这一含义。

独立审查曾发现并修复三类具体问题：ZIP实际解压长度不能只依赖文件声明；同一时间指数对应的多态矩阵元不能靠有限差分噪声“获得”可辨识性；通过预检的Z3估计量设置必须完整传递给生成任务。相应反例需要持续保留为回归测试。

真实科学计算仍需验证实际二进制与action、源到传播子的数值定义、残差解释、完整顺序源收缩、独立组态与源身份、流与极化约定，以及有限体积、格距、热态和激发态等系统误差。完整数据库哈希只证明对应字节未改变；`validated-numerics`只描述给定假设下的数值检查，不自动确认物理结论。
