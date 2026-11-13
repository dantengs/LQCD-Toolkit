# Contributing

Thank you for helping improve LQCD Toolkit. Contributions should make calculations easier to inspect and reproduce while preserving explicit conventions and scientific limits.

## Propose a concrete change

Use an issue or pull request to describe the affected operation, the expected behavior, and the observed behavior. Include a small reproducible example when possible. For a numerical issue, provide the relevant shapes, dtype, units, operator/mass ordering, time mapping, and tolerance. Synthetic arrays are usually sufficient; do not upload private gauge fields, credentials, operational logs, or machine-specific configuration files.

Do not report a numerical software defect solely from a variable name or a familiar-looking formula. State the assumptions under which the result should hold. Distinguish a confirmed implementation error from an unverified external-action convention or an unresolved physical interpretation.

## Development and tests

```sh
python -m pip install -e .
python -B -m unittest discover -s tests -v
```

Run tests relevant to the change. Tests should exercise an independent identity, a round trip, a minimal failure, or a controlled end-to-end example rather than merely restating the implementation. External-backend tests should use small process-contract doubles by default; they must not launch production calculations as part of ordinary test discovery.

Preserve original input data and investigation artifacts. Do not add recursive cleanup or deletion of user data to analysis commands or fixtures. A failed scientific operation should report the failure and retain inspectable outputs rather than silently dropping failed measurements or resamples.

## Changes to scientific behavior

For changes to axes, mass ordering, gamma/tensor definitions, Fourier phases, source normalization, time orientation, or current projection:

1. Write the old and new mathematical contracts explicitly.
2. Show the index or normalization derivation and a controlled numerical comparison.
3. State which existing outputs or profiles are affected; version the profile when its meaning changes.
4. Update both English and Chinese convention documentation.
5. Preserve any counterexample that motivated the change as a regression test.

For statistical changes, describe the independent sampling unit, source averaging, missing-data policy, configuration order, covariance normalization, resampling pairing, and identifiability checks. Do not interpret source measurements as independent gauge configurations. Do not claim that a converged optimizer or an estimated covariance establishes physical validity.

For a new external backend, document the executable interface, source and output layouts, action-specific transformations, residual definition, dependency provenance, and failure checks. Keep actual-action validation status separate from successful process execution. An external backend must not silently fall back to a mathematical reference operator.

## Data handling and decoders

Changes to the store must preserve original bytes and their identities. New numeric formats need explicit profiles and bounded parsing. Do not add general-purpose `pickle.load`, `pickle.loads`, or `Unpickler` execution for untrusted archived inputs. Unknown input should be rejected or retained as opaque data, not guessed from its filename.

For compressed formats, bound actual decompression output and verify stream completion; metadata-declared sizes alone are insufficient. Keep raw-preservation integrity, successful numeric decoding, and scientific metadata confirmation as distinct results.

## Public release hygiene and licensing

Keep examples portable: use relative paths and clearly marked placeholders. Do not commit personal names without consent, private email addresses, access tokens, SSH aliases, deployment hosts, private repository links, operational root paths, or institution-specific account information. This repository is for new implementation code, tests, synthetic fixtures, and templates; do not add historical source archives or original repository history.

The repository is distributed under the MIT License. Contributions must be material you are entitled to contribute under compatible terms. Preserve applicable notices and attribution. Do not add historical source bundles or vendor implementations. New format adapters should reuse the shared database/decoder/statistics core rather than import old scripts or create a second analysis stack. AI assistance should be disclosed accurately when relevant; it does not replace authorship provenance or technical review.

## Pull request checklist

- Describe the concrete problem and resulting behavior.
- Include the minimum reproducible evidence and relevant test results.
- Explain any changed scientific assumption, numerical tolerance, or file/profile contract.
- Update English and Chinese documentation where the user-facing behavior changes.
- Check that code, examples, fixtures, and templates contain no private deployment information or historical source bundles.
- State remaining limitations without presenting software tests as production physics validation.

## 中文贡献说明

欢迎提交可复现的问题和修改。请说明触发条件、实际行为、预期行为及最小示例。数值问题应包括数组形状、dtype、单位、算符和质量顺序、时间映射与容差；尽量使用合成小数据，不上传私人数据、凭据、运行日志或机器专用配置。

修改物理约定时，先写清修改前后的数学契约和指标推导，再提供独立验证，必要时升级profile版本，并同步中英文说明。统计修改需要明确独立样本单位、源平均、缺失数据、组态顺序、协方差与重采样配对。不要把源测量当成独立组态，也不要以优化器收敛替代科学验证。

普通测试不得启动生产求解任务。保留原始数据和失败现场，不加入自动递归删除；不静默剔除拟合失败的重采样。解码器采用明确且有资源限制的语法，不引入任意pickle执行；压缩数据必须检查实际解压长度和完整性。

提交前检查源码、示例、合成样例与模板，移除私人联系方式、主机和账号信息、SSH别名、私人仓库链接及运维路径。本仓库不接收旧源码归档或原仓库历史；新增格式适配复用公共读取与分析核心，不导入历史脚本。只贡献自己有权按兼容许可发布的内容，保留原作者署名和适用许可；如使用AI辅助，应准确说明，但仍需完成来源核对与技术审查。
