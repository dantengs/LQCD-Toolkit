# Release validation record

Date: 2026-11-14.

The implementation prepared for this release passed **136 unittest cases** on Python 3.14.6, NumPy 2.4.6
and SciPy 1.18.1. The suite includes the complete small reference workflow,
version-1/version-2 SQLite checks, newly implemented existing-data format adapters, correlated joint fits,
analytic Jacobian/rank regressions, effective-energy diagnostics and bounded
numeric decoder tests. It uses synthetic public fixtures. The external backend
process test uses a clearly identified test double, not a physical inverter.

The publication scope is the new general-purpose implementation, its tests,
synthetic fixtures, templates, and documentation. No historical source archive,
original repository history, vendor source tree, or real research dataset is
included. The modules for effective-energy diagnostics and CF-format
compatibility are new code, with one shared database reader and numerical core.
Old archives and original history remain private.

The recorded test run covers the implementation copied into this release
preparation tree. Re-run the commands below after packaging or modifying the
tree. A privacy check for a previous packaging layout is not evidence that a
later layout has passed; use the current public-tree checker and its actual
output before publishing.

Reproduce the checks:

```bash
python -m pip install -e .
python -B -m unittest discover -s tests
python tools/check_public_tree.py
```

Continuous integration also checks supported Python versions on a Linux runner.
Its result is independent of this local record. This validation does not
establish real external-action correctness, tuned physical inputs, statistical
independence, renormalization or a published decay form-factor result. See
[SCIENTIFIC_VALIDATION.md](SCIENTIFIC_VALIDATION.md) for the precise boundaries.

## 中文说明

本次发布准备采用的实现已通过 **136 项 unittest**，包含 v1/v2 数据库、共享 CF
格式适配、完整数学示例、相关联合拟合、解析 Jacobian/秩、有效能量及受限解码测试。
外部进程测试使用明确标注的替身，不是物理求解器。

本仓库只发布新通用实现、测试、合成样例、模板与文档，不发布旧源码归档、原仓库
历史、vendor 源码或真实科研数据。打包或修改后应在当前发布树重新运行上述测试
与公开内容检查；旧打包布局的检查不能替代当前树的验证。真实外部 action 的数值
正确性仍未验证，`external-numerics-unverified` 边界不变。
