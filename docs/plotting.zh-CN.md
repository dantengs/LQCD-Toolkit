# 可选科研绘图模块

安装：`pip install -e '.[plot]'`。只使用分析功能时不需要 matplotlib；
绘图模块也仅在真正绘制时加载它。无显示器环境可设置 `MPLBACKEND=Agg`。

本模块只展示已有结果，不计算有效质量、不拟合、不平滑、不合并源、
不转换单位，也不推导 z 或误差带。复杂数需先在分析层显式取实部、虚部
或其他有物理意义的投影，并在标签中说明。非有限值和形状不匹配会报错，
不会自动删除或补零。

## 接口

- `Series(x, y, label, x_unit, y_unit, yerr=None, uncertainty=None, level=None, samples=None)`：
  点估计及可选误差。`yerr` 是偏差，不是区间端点；支持 `(n,)` 对称误差
  或 `(2,n)` 的下、上偏差。
- `Band(x, lower, upper, label, x_unit, y_unit, kind, level=None, samples=None)`：
  `lower/upper` 是绝对上下界，x 必须严格递增。仅连接给定网格，不平滑，
  不外推，也不将上下界中点伪装成中心预测。
- `plot_series(series, xlabel=..., ylabel=..., bands=...)`：通用绘图。
- `plot_effective_energy(series, ...)`：有效能量；有效质量用 `ylabel="Effective mass"`。
- `plot_form_factors(series, ...)`：形状因子对 q²。
- `plot_z_expansion(series, bands=..., ...)`：展示外部算好的展开及区间；不做拟合或变量变换。
- `plot_correlator(data, fit, bands=..., ...)`：关联函数、已有拟合和残差；
  fit 必须在与 data 完全相同且递增的 x 网格上求值，不能携带 yerr。
- `export_figure(fig, stem, formats=("pdf", "svg", "png"), dpi=300)`：导出图及来源侧车。

误差类型必须声明为 `sd`、`sem`、`jackknife_se`、`bootstrap_se` 或
`confidence_interval`；最后一种需声明 `level`，如 `0.68`。类型会写入图例。
Band 的 `kind` 是 `confidence`（拟合量置信区间）、`prediction`（新观测预测区间）
或 `spread`（不声明覆盖率的离散范围）。前两者必须指定 `0 < level < 1`，
后者不能指定 level。逐点区间还是同时区间需在公开标签或图注中解释。
点误差不能自动变成拟合带，置信区间也不能替代预测区间。

关联函数残差定义为 data − fit，误差条复用原数据误差，视拟合为固定参考。
它**不是**考虑数据与拟合相关性后的残差估计量误差，图例和侧车均说明这一点。
给定预测点之间用直线连接，不生成额外预测点。

必须指定单位；无量纲用 `"1"`。同图的单位字符串必须完全一致，需调用者
先完成换算。模块不执行物理量纲代数。`xscale="log"` 或 `yscale="log"`
会拒绝非正点、误差端点或带下界，不会默默隐藏。

## 样本与可追溯性

所有绘图函数支持 `figsize_inches=(宽, 高)`，单位英寸，必须为有限正数；
默认 `(5.2, 3.6)`。可在生成时直接指定期刊列宽，避免排版时整体缩小导致字体过小。
例如 86 mm 宽、70 mm 高的单栏图（沿用英文示例中的人工数据 points）：

```python
fig = plot_effective_energy([points], figsize_inches=(86 / 25.4, 70 / 25.4))
fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)
export_figure(fig, "output/single_column_demo")
```

侧车记录实际导出尺寸，包括调用 `fig.set_size_inches(...)` 后的尺寸。
默认正文为 9 pt，图例为 8 pt；仍需按最终大小检查长标签和图例是否能容纳。
此示例是可选的单栏尺寸，不代表所有期刊的统一要求。
图例将误差枚举显示为易读的 “jackknife SE”“bootstrap SE”等，侧车保留原枚举名。

`SampleInfo(unit, count, grouping)` 的 unit 可为 `configuration`、`bin` 或
`demonstration`。grouping 说明组态内源平均、分箱宽度及独立性依据。
顺序源文件、传播子和噪声副本不能算独立组态样本；应先在组态内平均，
再考虑自相关和分箱进行重采样。绘图不验证独立性。`samples=None` 表示
未声明样本信息，不表示模块推断了 n；投稿前应补充说明。

映射现有分析输出时，可使用统计模块给出的均值与均值协方差，标准误为
其对角元的平方根，保留原组态/分箱定义；用现有模型和 `FitResult.parameters`
在指定 x 求值后构造 fit `Series`。图不能替代拟合诊断。

侧车 `stem.provenance.json` 保存数组形状和规范 float64 数值 SHA-256、
标签、单位、误差种类、分组、版本、样式及输出文件哈希，不保存输入输出路径
或原始样本编号。侧车不能替代源数据，应另行保存数组。文本字段原样写出，
请勿在其中放私有路径或身份信息。修改数据后应重新绘图以保持侧车一致。
PDF/SVG 保留可编辑文字；样式设置只在局部生效。不同依赖版本的输出不保证
逐字节相同，格式元数据可能含时间戳。

## 示例和命令行

完整 Python/JSON 示例见 [英文接口说明](plotting.md)。命令行：

```text
lqcd-plot spec.json output/figure
python -m lqcd_analysis.plotting spec.json output/figure
python tools/plotting_demonstration.py output/demo
```

JSON 顶层 kind 为 `series`、`effective_energy`、`form_factors`、`z_expansion`
或 `correlator`。关联函数用 data/fit，其余用 series 数组；bands 是可选数组，
formats 选择导出格式，其余键对应 Python 参数，未知键会报错。

随附演示全部使用明确标注的人工数据，仅测试显示流程，不可混入论文证据。
真实结果投稿前必须检查科学结论、拟合范围、误差定义、独立样本分组与图注。
