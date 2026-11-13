# LQCD Toolkit 使用指南

## 一份物理配置，一条运行命令

程序把物理输入、数据格式和机器环境分开。普通使用者填写系综、夸克质量、
初末态算符、流、动量、源位置和时间间隔；后端配置只需由了解求解器的人接入一次。
代码不内置任何私人账号、集群地址或生产目录。

```bash
python -m pip install -e .
python tools/create_reference_example.py examples/my_reference
python run.py check examples/my_reference/physics.json
python run.py run examples/my_reference/physics.json
```

示例会实际完成源生成、三条普通传播子求解、顺序源生成、顺序传播子求解、
两点/三点收缩和 SQLite 封存。它使用小型显式矩阵验证线性代数与数据流，
不使用真实规范场，不代表某种生产费米子 action。只有一个组态，故不自动拟合。

真实任务复制并填写 `configs/physics.template.json`，将 `template` 改为 `false`：

```bash
python run.py check my_physics.json
python run.py run my_physics.json
```

检查命令不启动求解、不建立输出。运行目录必须是新目录；程序不覆盖或清理旧结果。

## 代码分别负责什么

| 模块 | 用途 |
| --- | --- |
| `generation.py` | 点源/Z3 源、求解器对接、顺序源、CPU 参考收缩 |
| `workflow.py` | 根据物理配置预检、串联各阶段，可选继续分析 |
| `writer.py` | 关联函数直接写入 SQLite，逐条回读、验证选择清单后封存 |
| `database.py` | 只读打开版本化数据库，验证来源、分块顺序、长度和 SHA256 |
| `decoders.py` | 按显式 profile 读取数值 pickle、NPY、NPZ 和文本；不执行序列化函数 |
| `selection.py` | 核对组态/源、字面质量、算符、动量、时间方向 |
| `legacy_profiles.py` | 新写的 CF 格式兼容适配，复用同一读取、解码、选择与分析核心 |
| `statistics.py` | 源平均、组态身份对齐、分箱、jackknife、共享 bootstrap |
| `fitting.py` | 相关拟合、协方差截断、参数秩、完整 jackknife 拟合 |
| `analysis.py` | 数据库到周期两点函数拟合 |
| `joint_analysis.py` | 初态两点、末态两点、三点的多态联合拟合与解析 Jacobian |
| `diagnostics.py` | log/cosh 有效能量及完整 jackknife 传播 |
| `physics.py` | 谱模型、动量/q²、有限范围的 P→P 代数提取和显式 z 级数 |

公共读取器支持 `lqcd.opaque-chunks.v1/v2`，自产写入器输出 v1。既有 writer 的名称、时间和质量标签差异通过新写的 profile 适配，不包含或执行历史脚本，也不另写一套求解、重采样或拟合框架。存储细节见 [STORE_FORMAT.md](STORE_FORMAT.md)，
物理定义见 [中文约定](PHYSICS_CONVENTIONS.zh-CN.md)。

## 物理配置怎样填写

| 字段 | 含义 |
| --- | --- |
| `ensemble` | 明确的系综身份，不从文件名猜 |
| `geometry` | 整数 `nt,nz,ny,nx` |
| `quark_masses` | 夸克标签到正质量十进制字符串，单位为格点单位 |
| `process.*_quark` | 初态活跃线、末态活跃线、旁观线的显式对应 |
| `initial_operator/final_operator/current` | 所声明欧氏 gamma 基底下的字面算符 |
| `momentum_initial/final` | 整数 Fourier 模式 `[nx,ny,nz]` |
| `separation` | 满足 `0<T<NT` 的时间间隔 |
| `sources.positions` | `[x,y,z,t]`；本版同组态同源时间只支持一个源测量身份 |
| `configurations` | 不带前导零的唯一组态编号，以及对应规范场 |
| `backend` | 内联配置，或一次配置好的后端 JSON 路径 |
| `max_memory_bytes` | 此次参考计算允许使用的内存 |
| `analysis` | `null`、两点拟合或两点/三点联合拟合请求 |

数据路径相对于物理配置文件解析；独立后端文件中的路径相对于后端文件解析。
程序会在启动前核对全部任务的输入和求解器模板。检查通过仍不等于 action、
重整化、拟合窗口或物理近似已被证明正确。

点源支持已声明的非零动量。Z3 网格必须给出 `base_seed`、可整除空间边长的
网格间距，并声明 `grid_estimator="zero-momentum-source-average"`；本参考版本
只支持初末态均为零动量的网格估计，并除以网格支撑点数。网格点不是独立组态。

## 接入求解器与跨机器运行

先补齐 `profiles/xqcd.template.json` 与 `generation/xqcd_input.template`。
后端需要明确的可执行文件/启动器参数、action 依据、依赖文件、残差日志格式、
完整残差数量及容差。程序检查标准输入和大端 float64 输出，保存依赖、输入、
日志和输出校验值；不会猜生产参数，也不自动登录或提交作业。

机器差异应放入独立配置与作业包装层：本地、MPI 或调度器只选择适当的一层启动。
请在已分配的计算资源运行。参考收缩保留全张量，默认 1 GiB 内存预算会拒绝大格点；
它没有宣称具备生产级流式/MPI/GPU 性能。

外部运行仍标记 `external-numerics-unverified`，直到维护者独立核实二进制版本、
`read_source` 行为、传播子接触项/归一化及日志残差的物理含义。
退出码为零、文件长度正确和解析/合成测试通过，均不能代替真实后端科学验收。

## 从数据库分析

新生成的关联函数自动获得选择清单。旧数据需要补齐 writer、组态身份和物理约定。
无法确认的内容可以检查原始字节，但不会被自动解释成某个衰变通道。

```bash
python run.py list run-output/correlators/data.sqlite --limit 10
python run.py inspect run-output/correlators/data.sqlite --id 1 --profile savedata.pickle.numeric.v1
python run.py verify run-output/correlators/data.sqlite --all-payloads
python run.py fit2 my_fit.json --output c2-result.json
python run.py joint my_joint_fit.json --output joint-result.json
```

两点拟合配置版本为 `lqcd.fit2.v1`，明确数据库/选择清单、组态顺序、bin 大小、
固定时间窗口、协方差截断、态数、初值和边界。模型是同算符源汇的正周期对角谱；
实部投影必须有物理依据，不能自动取绝对值或按信噪比挑实虚部。

`generation/joint_fit.template.json` 展示联合拟合配置。矩阵元轴为
`M[final_state,initial_state]`，C3 在显式给出流相位/倍率之前保留复数。
新数据选择 `initial-forward` 物理时间；旧顺序源数据必须选择其确切时间 profile。
三点窗口排除接触端点。解析 Jacobian 用来识别精确简并；单个 T 下多态矩阵元
可能无法分别确定，程序会拒绝给出虚假的有限误差。

统计上先在每个组态内平均源，再对组态等权；通道按身份对齐，按指定链顺序等大小
分箱且不丢余数。所有 jackknife 样本使用同一预选窗口和全样本均值协方差，
任何样本失败都明确终止。不同系综不能靠同一行号混为一组联合重采样。
普通 p 值仅是渐近诊断，不包含数据择窗、协方差估计和物理模型误差的完整校准。

## 有效能量和形状因子

有效能量支持 `log(C[t]/C[t+1])` 与局部 cosh 关系，要求正实有限输入和有效定义域。
jackknife 版本不截断 acosh 参数、不 abs、不删除失败样本，并报告失败的组态身份。

P→P 工具只接收已经确认相位并重整化的实 Minkowski 流和连续体 on-shell 运动学。
它避免除以 q²，保留 `f0(0)=f+(0)`，并拒绝零反冲秩亏。它不是带噪流的相关拟合器，
不自动转换欧氏相位，也不自动提取 P→V/V→P 形状因子。z 工具是显式带极点的多项式，
不默认宣称 BCL/BGL。完整边界见物理约定与科学验证文档。

## 仓库范围、许可与贡献

本仓库仅包含新通用实现、测试、合成样例和模板。有效能量诊断与 CF 格式兼容适配均为新代码；读取既有格式不意味着包含其历史 writer。旧源码归档与原仓库历史均保留私有。私人部署信息、真实科研数据和第三方/vendor 源码不随本仓库发布。
MIT 许可不替代外部依赖自己的许可。贡献署名和规则见
[CONTRIBUTORS.md](../CONTRIBUTORS.md) 与 [CONTRIBUTING.md](../CONTRIBUTING.md)。
