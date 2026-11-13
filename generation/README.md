# LQCD Toolkit：生成与联合分析接口

本文描述新写的通用实现，不依赖公开旧脚本或历史源码归档。普通命令入口见[使用指南](../docs/USER_GUIDE.zh-CN.md)，详细公式见[物理约定](../docs/PHYSICS_CONVENTIONS.zh-CN.md)。

这一版本提供可执行的本地参考流程：生成源 → 求三个质量的传播子 → 生成 sequential source → 求初态质量的 sequential propagator → 收缩 C2(initial)、C2(final)、C3。联合分析从封存的 SQLite 读入这些通道，按配置 ID 对齐、先平均源、再分箱，做同一协方差下的固定窗口拟合。

普通使用者填写根工作流的 `configs/physics.template.json`；本目录的 `generation.template.json` 用于直接调用下层生成接口。两者都明确几何、组态、源位置、三条质量线路、初末态算符/动量、流和时间间隔。后端安装人员另提供经过核实的 XQCD 可执行文件和输入模板；这是一次后端接入工作，不应让使用者猜 action、残差日志格式或传播子归一化。根命令入口可以直接调用下述两个高层函数，不需要暴露内部张量轴。

## 生成接口

```python
from lqcd_analysis.generation import run_generation, correlator_records
product = run_generation(config, "new-run-directory")
records = correlator_records(product)  # 交给 create_generated_store，无逐通道小文件
```

`run_generation` 接受字典或 JSON 路径，输出目录必须不存在。`stage` 可为 `source`、`solve`、`contract`、`full`；后两种分段执行方式通过 `input_product` 指向前阶段目录。source→solve→contract 与 full 使用相同算法。`solve` 包含三个普通传播子及 sequential source/propagator，避免把不闭合的顺序源当作完整三点计算。输出是一个 `arrays.npz`、一个 `product.json`，外部求解另保留每次调用的输入、日志及 backend 必需的源/传播子文件，不删除、不覆盖。

所有传播子轴由 `xqcd-be-f64.v1` 的实现固定为 `[source_spin=4,source_color=3,sink_spin=4,sink_color=3,t,z,y,x]`。磁盘 raw 格式在 source_color 后插入 `[real,imag]`，大端 float64，无 header。用户不需要手填轴或字节序。

源有两种：

- `{"kind":"point","position":[x,y,z,t]}`：12 个 spin/color-diluted 单位点源。
- `{"kind":"z3-grid","position":[x,y,z,t],"grid_spacing":[dx,dy,dz],"seed":123}`：PCG64 可复现 Z3 空间网格；spacing 必须整除对应边长。RNG 流不声称等于旧 XQCD RNG。当前高层网格收缩仅支持零动量，必须声明 `grid_estimator="zero-momentum-source-average"`，收缩除以网格点数。非零动量使用点源；本版没有自行猜测动量网格的源投影。单个噪声实现保留交叉项，统计误差需要跨独立组态/噪声处理，不能把网格点当独立组态。

质量命名固定：`initial=m1`（sequential inversion），`final=m2`（被共轭的普通传播子），`spectator=m3`（顺序源输入）。C2 的 ordered masses 为 `(m3,m_active)`，C3 为 `(m1,m2,m3)`。算符和 gamma 基底使用 `cpu-sequential-reference.v1` 中明确给出的定义，保留复数输出，不做自动 abs 或实虚部选择。特别地，`sigmaij` 定义为 `gi.gj=[gi,gj]/2`，**没有额外 i**；`gamma('sigma12')` 遵循这个参考标签约定，不把另一种常见的 `i[gi,gj]/2` 定义混进去。

CPU ST 时间：普通源位于 raw x0，sequential source 位于 raw `(x0+T)%NT`，插入初态 Gamma1；输出 C3[j] 取 raw `(x0+T-j)%NT`。所以 j=0 是初态、j=T 是末态，沿 raw 时间反向。新记录用 `time_convention="initial-forward"` 表示 j 是物理初态到流的 tau，并另外保存 raw 映射；这与旧 selection 的 base/source-forward 不同。

CPU 参考实现保留多个全张量，不能默认在登录节点运行 32³×64。预检估计最多 16 个传播子张量的工作内存，dense 参考求解另加矩阵工作区，默认限 1 GiB；大格点必须明确给计算节点内存预算。这里没有声称生产级流式/MPI 收缩性能。

## 两种求解后端的边界

`explicit-matrix-reference` 接收每条质量线路的显式数值 D 矩阵 NPY，解 `D X=B`，逐 RHS 核 `||DX-B||/||B||`，限矩阵维数 4096。必须声明 `scientific_mode="mathematical-validation"`；它检验线性代数、轴、顺序源依赖和收缩流程，**不是现有 overlap action 的替代实现**。

`xqcd.overlap-inverter-stdin.v1` 使用真实入口契约 `argv + stdin`，`shell=False`，不生成 qsub/condor 作业，不修改环境，不自动 SSH。`argv` 可显式包含已配置的 MPI launcher 与可执行文件。输入模板只替换 `$source,$workdir,$gauge,$config_id,$mass,$nt,$nz,$ny,$nx`。校验 geometry、单组态、单质量、`read_source`、无附加 momentum/slice 参数及输出路径。`Hwilson_name/ov_evecname/prop_name` 必须位于新的 workdir；需要的已知缓存文件由 `dependency_files` 显式复制到相应相对路径，原始缓存不修改。模板中的 REQUIRED 值必须由后端负责人落实，不能根据外部程序名称自动推断当前 action。

已确认的源输出规则是 `${source}.read00.m{mass:.6f}`，大端 double。后端必须给出有出处的残差正则表达式（一个数值捕获组）、残差定义、预期数量（至少 12）、阈值；缺残差/输出/非有限数/非零退出状态全部终止。**不能仅根据某个 `check_residual` 输入项的名称，就将日志解释为传播子真实残差。** 外部二进制与其源码版本的一致性、`read_source` 内部处理、`one_minus_halfD` 接触项和归一化仍需真实后端验证。日志抽取值的物理意义来自显式后端契约，并不因 exit code=0 自动成立。

`backend_status` 永远区分 `explicit-operator-reference` 与 `external-numerics-unverified`。尚无真实 gauge/完整生产传播子，未宣称生产物理结果重现。

## 联合分析接口

```python
from lqcd_analysis.joint_analysis import fit_joint_c2_c3
result = fit_joint_c2_c3(database, selections, settings)
```

`selections` 的键固定为 `initial,final,three_point`。C2 保留显式 real 投影，C3 保留 complex，直到 `current_projection={"phase":[re,im],"scale":...,"evidence":...}` 执行 `real(phase*C3)*scale`。phase 要求单位模，scale 非零。这里输出的是声明约定下的实矩阵元，不自动改为 Minkowski、不自动给 P→V/V→P 形状因子。

`joint_fit.template.json` 展示配置。多态数各 1–4，能量用有序正 gap，overlap 为正幅度，矩阵 `M[final_state,initial_state]` 使用显式有限边界。C2 要求对角正谱；C3 是 forward-only，无 thermal wrapping/contact 项，所以窗口严格排除 0 与 T。相同初末 C2 需要专门共享参数的模型，本版拒绝把它重复当独立 sector。

从新生成数据选 `c3_time_mapping="initial-forward"`；旧 CPU ST 选 `"cpu-st"`。后者接收原始 sink-backward j，或将已经转换成 base-source-forward 的数组明确按 `(T-tau)%NT` 还原一次，并记录转换。所有通道先核同 ensemble/NT/质量线路/算符/动量；按配置集合 exact 或显式 intersection 对齐，丢弃项写结果。源平均后等配置分箱，余数报错。全样本与所有 JK 共享已选窗口和全样本均值协方差，任何样本失败/不可辨识立即停止。普通 pvalue 只作为渐近诊断。

## 可核对的证据与测试

新实现的数学契约和可复现反例见[科学验证说明](../docs/SCIENTIFIC_VALIDATION.md)，当前测试记录见[验证结果](../docs/VALIDATION_RESULTS.md)。本仓库不分发旧源码、历史归档或原仓库历史；外部求解器仍需独立版本与 action 验证。

`test_generation.py` 覆盖 Clifford 恒等式、点/Z3 源、raw 轴往返、显式 D 真实残差、非零动量与绕回顺序源、独立逐指标 CPU 收缩、分阶段/完整参考闭环、真正启动的 **测试替身** 后端契约及缺失残差硬失败。`test_joint_analysis.py` 从永久保留的 sealed SQLite 合成 fixture 验证不等源数下联合参数恢复、时间转换恰好一次、多态 M 轴序、跨集合/ensemble/质量/相位/接触窗口拒绝及拟合失败传播。测试替身不是 XQCD；合成 fixture 不是科研数据。没有运行旧脚本，也没有自动清理测试产物。
