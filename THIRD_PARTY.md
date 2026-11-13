# License boundaries and external dependencies

The root MIT license covers the new first-party implementation, tests,
synthetic fixtures, templates, and documentation distributed in this repository.
It does not relicense external software or data. Historical source archives and
the original repository history remain private and are not included here.

NumPy and SciPy are installed as dependencies and retain their respective
licenses. The external inverter is supplied and licensed separately by its
operator; this repository includes an input/output adapter, not its binary or
source tree.

No vendor source tree or historical source bundle is included. Obtain any
external solver and its dependencies separately from their upstream projects,
under their original licenses. Compatibility with a file format does not imply
ownership of its original writer or permission to redistribute that writer.

`lqcd_analysis.legacy_profiles` and `lqcd_analysis.diagnostics` are new
implementation modules. The former adapts existing-data labels and layouts
through the shared storage/analysis core; it does not contain or execute old
project scripts. The latter implements explicit effective-energy diagnostics.

No production gauge field, propagator, correlator sample, private machine
configuration or credential is distributed. Public parser fixtures and the
reference example are synthetic and explicitly labeled.

## 中文说明

根目录 MIT 许可覆盖本仓库分发的新实现、测试、合成样例、模板与文档，不改变外部
软件或数据的许可。旧源码归档和原仓库历史保留私有，不随本仓库发布。

NumPy、SciPy 和使用者另外安装的求解器保留各自许可。这里提供求解器接口，不提供
外部求解器二进制或源码。`legacy_profiles` 与 `diagnostics` 都是新模块；兼容既有
数据格式不等于包含、执行或取得其历史 writer 的再分发权利。公开样例均为合成数据，
不包含真实规范场、传播子、科研关联函数或私人部署配置。
