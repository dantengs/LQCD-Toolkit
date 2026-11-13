# Existing-data format compatibility / 既有数据格式兼容

`legacy_profiles.py` is newly written compatibility code. No historical writer scripts or source archives are distributed or executed by it.

The public numerical core owns byte decoding, database access, source averaging,
resampling and fitting. `lqcd_analysis.legacy_profiles` adds a narrow writer
adapter for per-configuration dictionaries. It contains no machine paths,
scheduler commands, solver defaults or physical mass assignments.

`legacy_profiles.py` 是新写的数据格式兼容代码，不包含或执行历史 writer 脚本，也不分发旧源码归档。

公共数值核心负责字节解码、数据库读取、源平均、重采样和拟合。此模块只核对既有
writer 的名称、数组布局、标签与显式身份。计算机器、路径、求解器和队列仍由独立
运行配置提供，不能用机器名称猜测物理约定。

## Supported format and explicit evidence / 支持格式与证据

The supported container is a dictionary from literal channel keys to real numeric
arrays of shape `(2, NT)`: row 0 is the real part, row 1 the imaginary part.
The existing bounded `savedata.pickle.numeric.v1` decoder handles its bytes;
no pickle callable or old project script is executed by this adapter.

The basename grammar is:

```text
cf<configuration>_t<source>_<mass1>_<mass2>.bin
cf<configuration>_t<source>_T<separation>_<mass1>_<mass2>_<mass3>.bin
```

Configuration IDs remain strings: `0007` and `7` are different declared IDs.
Masses use `Decimal` comparison, so the filename `0.250000` can match a key's
`0.25` exactly. Labels must be finite and positive; this is syntax validation,
not a claim that a positive number is the intended physical mass. Physical
mass ranges and solver input semantics belong to the generating configuration.

文件名不选择时间方向。调用者必须明确给出 `WriterProfile` 和结构化
`WriterEvidence(version, source)`；证据应指向实际使用版本的 writer 或运行记录。
非空证据字段只能记录调用者声明，程序不能自动证明它真实对应某次历史运行。

| Explicit writer profile | Absolute lattice index of stored entry `t` |
|---|---|
| `two-point.forward.v1` | `(source_time + t) % NT` |
| `three-point.source-backward.v1` | `(source_time - t) % NT` |
| `three-point.sink-backward.v1` | `(source_time + separation - t) % NT` |

`stored_to_absolute_times` returns this mapping. **Array values stay in stored
order.** It does not roll, reverse or re-normalize the data. These are writer
layout versions, not universal assertions about all old two/three-point files.
Three-point files require explicit integer `0 <= separation < NT`; source times
must be integer `0 <= source_time < NT`. Booleans are rejected as times/extents.
The fit model remains responsible for an appropriate physical separation and
window, including any coincident-time exclusions.
Allowing `T=0` here approves only the time-index arithmetic. It does not validate
coincident-time insertions, contact terms or their treatment in a physical fit.
NT is bounded by the shared numeric decoder's `DEFAULT_LIMITS.max_dimension`
before a time-index array can be allocated.

三点两个反向 profile 的原点相差 `T`。即使文件名相同，改变 profile 也会改变
时间映射，故必须依据真实 writer 证据选择。数组本身不会被再次反转或平移。

## Literal key slots and physical identity / 标签槽位与物理身份

Supported channel key grammars are:

```text
<mass1>_<operator1>_<mass2>_<operator2>_<momentum>
<mass1>_<operator1>_<mass2>_<operator2>_<mass3>_<operator3>_<momentum1>_<momentum2>
```

Operators use a literal whitelist (`Id`, `g1`–`g5`, `g5.g1`–`g5.g4`,
`sigma12`–`sigma34`, with increasing tensor indices), or an explicit custom
`allowed_operators` table. Custom labels are compared as strings and never
evaluated as expressions. Neither mass size nor an operator slot automatically
identifies an initial state, final state, current or spectator.

Delimiter-free momentum tokens use the existing
`selection.parse_momentum_token` with a mandatory allowed-momenta table.
Negative components such as `00-1` are supported. A token such as `1110` fails
if both `[1, 1, 10]` and `[11, 1, 0]` are allowed: the adapter does not assume
single-digit components.

The scientific identity extends the core's channel identity with one
measurement's `config_id`, `source_id`, `source_time`, and explicit writer-slot
metadata:

- `ensemble`, `kind`, `nt`, `ordered_masses` and distinct `mass_roles`;
- `ordered_operators` and `operator_roles`, mapping source/sink/current to slot
  indices, together with the core's `source_operator`, `sink_operator`, and
  three-point `current` labels;
- `momentum_initial`, `momentum_final`; three-point `momentum_roles` explicitly
  maps initial/final to the two key momentum slots;
- three-point `separation`;
- `physics_status` equal to `confirmed` or `synthetic-validation`, plus a nonempty
  `physics_evidence` description.

These declarations are cross-checked against filename and key labels. Every
decoded CF key must have the same filename mass slots, a valid `(2, NT)` array,
and finite numeric values. Differently written keys that denote the same
Decimal masses, operators and momentum are rejected as semantic duplicates.
An identity declaration is evidence supplied by the caller, not an automated
physical certification.

必须显式指定夸克线与算符的角色。正质量、文件名或目录中的组态计数，都不能代替
对物理通道和统计样本身份的确认。缺少身份、角色或 writer 证据时，科学适配拒绝
继续；原始字节仍可由保存数据库无损读取。

## API and synthetic example / API 与合成示例

```python
from lqcd_analysis.legacy_profiles import (
    WriterEvidence, WriterProfile, adapt_cf, parse_cf_filename,
    parse_array_key, stored_to_absolute_times,
)

profile = WriterProfile(
    "two-point.forward.v1", 8,
    WriterEvidence("synthetic-writer.v1", "In-memory fixture specification"),
)
identity = {
    "ensemble": "synthetic-ensemble",
    "kind": "2pt", "nt": 8,
    "config_id": "0007", "source_id": "source-six", "source_time": 6,
    "ordered_masses": ["0.125000", "0.250000"],
    "mass_roles": ["active", "spectator"],
    "ordered_operators": ["g5", "g5"],
    "operator_roles": {"sink": 0, "source": 1},
    "source_operator": "g5", "sink_operator": "g5",
    "momentum_initial": [0, 0, -1], "momentum_final": [0, 0, -1],
    "physics_status": "synthetic-validation",
    "physics_evidence": "Synthetic role assignment; no physical result claimed",
}

# `decoded` should come from the core's hash-verified store.decode(...) using
# profile="savedata.pickle.numeric.v1". No new database reader is needed.
measurement = adapt_cf(
    decoded,
    filename="cf0007_t6_0.125000_0.250000.bin",
    profile=profile,
    array_key="0.125_g5_0.25_g5_00-1",
    identity=identity,
    allowed_momenta=[[0, 0, -1]],
    component="complex",
    stage="synthetic-validation",
)
```

`parse_cf_filename` and `parse_array_key` are pure label parsers.
`adapt_cf` accepts the core's `DecodeResult` and returns a `LegacyMeasurement`
with values, absolute-time indices, parsed labels, identity and provenance.
`decode_cf(payload, ...)` is a thin standalone convenience wrapper over the same
core decoder and adds the payload's SHA256. Database callers should use
`store.decode` and retain that store record's verified hash and original source
lineage; the adapter does not authenticate a database or infer file provenance.

No CLI or existing fitting entry point is silently changed by this module.
A caller connecting it to a fit must retain the existing core's configuration
alignment, source averaging, time-convention checks, binning and covariance
requirements. The result's values are one observation, not fitted physics.

## Shared statistics bridge / 复用公共统计接口

`to_measurement_channel(measurements, *, time_action="as-stored")` accepts a
sequence of validated `LegacyMeasurement` objects and returns the existing
`selection.MeasurementChannel`. The keyword `time_action` is required, and
`as-stored` is its only supported value. No second source-averaging or fitting
implementation is introduced.

```python
from lqcd_analysis.legacy_profiles import to_measurement_channel

# Every item has already passed adapt_cf/decode_cf with explicit identity.
# Include at least two declared configuration IDs, with distinct source
# measurements within each configuration.
channel = to_measurement_channel(validated_observations, time_action="as-stored")
samples = channel.config_samples()  # Existing equal-configuration statistics.
```

Observations must share the writer profile and evidence, NT, component, stage,
writer key slots and declared physical channel identity. Decimal-equivalent
mass spellings compare equally. Only `config_id`, `source_id` and `source_time`
are removed when comparing and forming the common channel identity. Other
per-observation metadata should remain in provenance, not be disguised as a
shared physical declaration.

The bridge rejects duplicate `(configuration, source_id)` pairs and duplicate
`(configuration, source_time)` writer measurements, even when a source is given
a different label. Multiple source measurements from a single configuration
do not meet its minimum of two declared configurations. Configuration count
does not itself prove statistical independence; source averaging, alignment,
binning and jackknife requirements remain the shared core's responsibility.

`channel.times` is the common **stored tau** axis `0, ..., NT-1`. It is not one
source's absolute lattice axis. Every observation's absolute mapping remains
in `channel.provenance["files"][i]["absolute_lattice_times"]`; normal and
sink-origin backward profiles cannot be mixed into one channel. The provenance
also declares `output_time_convention` so a downstream fit can reject an
unsupported time convention without silently reversing data.

`component="complex"` is preserved in the shared channel, but the existing
real-observable statistics deliberately reject `channel.config_samples()` for
complex arrays. Choose an explicitly justified `real` or `imag` component when
adapting observations before entering that statistics path. The bridge does
not pick a projection or a phase convention.

桥接复用公共 `MeasurementChannel.config_samples()`，不同源数目仍按“每组态一行”
平均；不会把来源更多的组态变成更多独立组态。不同 source 的绝对格点时间映射
只写入各自来源记录，公共横轴保持 stored tau。复数统计需要调用者事先明确选择
实部或虚部，不能由适配层隐式决定。

## All/Ave and exact memberships / 聚合数据的身份边界

Legacy All files can concatenate configurations along axis 1, producing
`(2, Nconfig * NT)`. Other writer modes can average configurations. Source
averages may preserve the configuration axis, while Ave files can additionally
combine gamma/momentum channels with signs and assign one combined array to
several keys. File counts, array shapes and directory labels alone cannot
recover these decisions or identify independent samples.

An old aggregation pattern only counts input files, then appends each key when
present. If one configuration is missing a key, that key can have fewer rows
than the advertised configuration count. Exact **per-key** membership and shape
checks are therefore necessary.

`validate_aggregate_membership(arrays, nt=..., exact_members=...,
expected_keys=...)` is a diagnostic validator with this declaration grammar:

```python
exact_members = {
    "channel-key": [
        {
            "config_id": "0007",
            "sources": [
                {"source_id": "source-two", "source_time": 2,
                 "sha256": "a" * 64},
            ],
        },
        # One output row per explicit configuration, in the actual saved order.
    ],
}
```

Every expected key needs its own complete declaration. The validator checks
identical key sets, configuration/source memberships and order, non-null IDs,
unique configurations, distinct writer source times, SHA256 syntax, finite
arrays, and exact `(2, Nconfig * NT)` lengths. Several source members in an
output row still represent one configuration. It does not verify that the old
writer actually used the declared members, nor define a derived-observable
mixing/normalization rule.

**Initial support deliberately stops at diagnostics for aggregates.**
`AggregateLayout.scientific_loading_supported` is always `False`, and
`adapt_cf`/`decode_cf` reject All/Ave/source-average stages even if membership
information exists. Scientific aggregate loading requires a future shared-core
contract for authenticated membership and per-key derivation. Without it, use
the preserved per-configuration CF data.

本版本不把聚合文件自动转换为新的独立样本，也不重写重采样或拟合。缺少 exact
members 时诊断立即失败；即使声明完整，仍不会把派生平均冒充原始测量。文本导出、
二进制副本、不同源位置及被复制的 Ave 键，都不能额外增加独立组态数。

## Verification / 验证

`tests/test_legacy_profiles.py` uses only in-memory synthetic NumPy dictionary
payloads. It covers Decimal labels, leading-zero configuration IDs, negative and
ambiguous momentum tokens, explicit slot roles, the three distinct time maps,
unchanged stored order, finite `(2, NT)` shapes, missing physical declarations,
per-key aggregate membership defects and the diagnostic-only aggregate boundary.
The shared-channel tests exercise two configurations with two sources each,
unequal source counts, duplicate/relabelled measurements, mixed profiles and
components, and the existing real-only statistics boundary for complex data.
