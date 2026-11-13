# LQCD Toolkit

**明确物理约定的格点 QCD 工作流：从源生成、传播子求解到介子关联函数与相关拟合。**

[English](README.md) · [使用指南](docs/USER_GUIDE.zh-CN.md) · [物理约定](docs/PHYSICS_CONVENTIONS.zh-CN.md) · [科学验证范围](docs/SCIENTIFIC_VALIDATION.md)

LQCD Toolkit 将一份物理配置连接到可追溯的计算流程：生成源，通过明确指定的后端求解传播子，构造并求解顺序源，收缩二点与三点函数，将自产关联函数直接写入 SQLite，并按需执行拟合。生成器同时输出带测量身份的 selection sidecar，用户无需手工收集数据库文件 ID。

本仓库仅包含新通用实现、测试、合成样例和配置模板。旧源码归档与原仓库历史均保留在私人存储中，不随本仓库发布。

对于已有的兼容数据库，也可以只读查询、检查、验证和分析。公共读取器支持 `lqcd.opaque-chunks.v1/v2`，自产写入器输出 v1。[CF 格式兼容适配](docs/LEGACY_FORMATS.md)将明确的名称、Decimal 质量标签与时间映射接入同一套核心。文件解码、测量选择、统计重采样与物理解释分别处理：文件字节完整不等于已经确定其物理通道。

**当前科学验证状态：**小规模显式矩阵工作流和数值分析已测试，并完成数学示例的端到端运行。项目尚未对真实外部 XQCD action 做数值验证。配置检查通过或外部程序成功退出，都不代表 action、重整化约定或物理形状因子已经得到认证；外部结果保留 `external-numerics-unverified` 状态。

## 已有功能

| 部分 | 功能 |
|---|---|
| 源生成 | 点源；可复现的自旋/颜色稀释 Z3 空间网格源；显式指定位置、格点尺寸和随机种子 |
| 求解器 | 带真实残差检查的小规模稠密矩阵参考求解；通过 stdin/read-source 接口连接外部 XQCD 求解器 |
| 顺序源流程 | 用 spectator 传播子构造顺序源，以初态 active 质量反演，执行 CPU 参考 C2/C3 收缩 |
| 存储 | 直接建立采用 `lqcd.opaque-chunks.v1` 格式的封存 SQLite，保存字节哈希、来源信息和自动选择清单 |
| 已有数据 | 只读列出与检查原始记录、完整性验证、按显式格式解码数值数组 |
| 统计 | 先在每个组态内平均源，再显式对齐组态、等大小分箱、jackknife；另提供配对 bootstrap 工具 |
| 拟合 | 相关多态 C2 拟合，以及初态 C2、末态 C2、C3 联合拟合；协方差 SVD 截断、联合模型解析导数、参数秩与失败检查 |
| 物理数值工具 | 明确 overlap 归一化的谱模型、动量与 Minkowski q²、受限的 P→P 矢量流提取、实 z 映射和简单带极点 z 级数 |

这是 CPU 参考实现，实际 Dirac action 由外部求解器提供。它不自动提取 P→V 或 V→P 形状因子；能够执行一般 gamma 收缩，不等于已经实现某个向量介子衰变通道的完整分析。

## 安装与开始使用

请使用 Python 3.10 或更新版本，并满足 `pyproject.toml` 中的依赖要求。数值运行依赖 NumPy 和 SciPy。

```sh
python -m pip install -e .
```

不准备规范场或外部求解器，也可以先运行完整数学示例：

```sh
python tools/create_reference_example.py examples/my_reference
python run.py check examples/my_reference/physics.json
python run.py run examples/my_reference/physics.json
```

该示例创建三份48×48参考矩阵，贯通源生成到SQLite输出。它只有一个组态，明确关闭统计分析，不代表物理QCD数据。示例目录和输出目录都必须是新目录。

将 `configs/physics.template.json` 复制为 `physics.json`，填写物理参数，并将 `template` 改为 `false`。模板保留的空项需要用户填写。使用外部求解器时，先复制并配置 `profiles/xqcd.template.json` 及其引用的求解器输入模板，再让物理配置引用这份后端配置；同一安装环境通常只需配置一次后端。

```sh
python run.py check physics.json
python run.py run physics.json
```

`check` 检查计划中的测量、路径、格点尺寸、后端接口和内存估计，不启动求解器。`run` 要求新的输出目录，并保留中间产物和来源记录。再次运行时选择新目录，不覆盖已有结果。

物理配置包括以下内容：

- 系综名称与明确的组态 ID；
- 格点尺寸、源位置；
- 夸克质量标签，以及有顺序的 initial active、final active、spectator 三种角色；
- 初末态插值算符、流算符、整数动量模式和时间间隔；
- 后端配置、内存预算，以及可选的分析设置。

下面是 **process 字段片段**，不是完整可运行配置：

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

这些标签只是选择已配置的输入，不承担质量标定或物理态鉴定。这个非零动量示例应使用点源。当前 Z3 网格估计量只支持**初末态动量均为零**，并要求显式填写 `grid_estimator="zero-momentum-source-average"`。

数学验证可以选择 `explicit-matrix-reference` 后端和 `scientific_mode="mathematical-validation"`，提供三种角色各自的算符矩阵。这种模式验证线性代数和数据流，不会将参考矩阵自动替换成真实格点 action。完整配置和输出结构见[使用指南](docs/USER_GUIDE.zh-CN.md)。

## 只读检查与分析已有数据库

以下命令读取兼容的封存数据库。拟合输出写到新的结果文件，数据库保持不变。示例 ID 仅用于演示，实际 ID 由 `list` 查询。

```sh
python run.py list results/correlators/data.sqlite --limit 20
python run.py inspect results/correlators/data.sqlite --id 1
python run.py inspect results/correlators/data.sqlite --id 1 --profile savedata.pickle.numeric.v1
python run.py verify results/correlators/data.sqlite --all-payloads
python run.py fit2 fit2.json --output fit2-result.json --verify-database-hash
python run.py joint joint.json --output joint-result.json --verify-database-hash
```

`fit2.json` 和 `joint.json` 是分析设置，不是用于生成数据的物理配置。它们引用明确的 selection 清单和固定拟合窗口。拟合前需要确定组态/源身份、时间映射、分量投影及协方差设置；自产数据的 selection 清单由程序自动生成。

也可以使用 Python API：

```python
from lqcd_analysis.database import CorrelatorStore
from lqcd_analysis.workflow import plan_workflow, run_workflow
from lqcd_analysis.analysis import run_fit_file
from lqcd_analysis.joint_analysis import run_joint_file

plan = plan_workflow("physics.json")  # 仅检查

with CorrelatorStore("results/correlators/data.sqlite") as store:
    for record in store.iter_files(limit=5):
        print(record.id, record.path, record.sha256)

# 需要执行时显式调用：
# run = run_workflow("physics.json")
# two_point = run_fit_file("fit2.json", verify_database_hash=True)
# joint = run_joint_file("joint.json", verify_database_hash=True)
```

## 会影响结果的约定

公开生成 profile 为 **`cpu-sequential-reference.v1`**。约定需要与数据来源一起保留：

- 原始传播子采用 big-endian float64 的实/虚分开存储，自旋与颜色的 source 轴在前。C2 质量顺序为 `(spectator, active)`，C3 为 `(initial, final, spectator)`。
- 顺序源在指定时间切片形成 `Gamma_initial × spectator`，乘负号初态动量 Fourier 相位，再以初态 active 质量反演。
- CPU 顺序三点函数的存储 `j` 对应 `(source_time + separation - j) % NT`；它沿递减的原始格点时间，从初态插入位置走向末态插入位置。
- 不同指标的参考张量标签使用 `sigma_ij = gamma_i @ gamma_j`，**没有额外的 i**。
- C3 保留复数，直到显式指定相位与归一化投影；不自动执行 Euclidean→Minkowski 转换。
- Z3 网格收缩除以非零支撑点数。它是源平均的随机估计量，单次噪声实现不等于精确的点源平均。

将新生成数据与历史数据结合前，请先阅读[物理约定](docs/PHYSICS_CONVENTIONS.zh-CN.md)。

## 适用范围与可复现性

稠密参考后端的最大维数为 4096，这里的维数是 `12 * NT * NZ * NY * NX`。张量工作流另有保守的内存估计与显式预算检查，默认预算为 1 GiB。常见生产体积的单个传播子就可能占用数 GiB；提高预算不会使本程序自动变成分布式或 GPU 收缩程序。

外部可执行程序与 MPI 环境由使用者提供。action 参数、边界条件、依赖文件和残差解释需要明确填写。程序检查退出状态、输出长度和所填写的残差契约，但这些检查无法证明该契约或实际二进制的物理正确性。

数值解码器只接受有限且明确的数值语法；不支持的格式不会退回到可执行 pickle 加载。SQLite 读取器对每次返回的原始记录验证长度与 SHA256，并可额外校验完整数据库哈希。复现科学结果仍需保留 action、质量、格点尺寸、源定义、流约定和统计假设。

## 文档与贡献

- [使用指南](docs/USER_GUIDE.zh-CN.md)：配置、命令、输出和各模块作用。
- [物理约定](docs/PHYSICS_CONVENTIONS.zh-CN.md)：轴、指标收缩、谱模型、形状因子和单位。
- [科学验证范围](docs/SCIENTIFIC_VALIDATION.md)：已测试的内容、已修复的反例及尚待实际后端验证的部分。
- [参与贡献](CONTRIBUTING.md)：可复现修改、测试、文档和公开数据注意事项。
- [贡献者](CONTRIBUTORS.md)：贡献署名，包括 AI 辅助。

`diagnostics.py` 与 `legacy_profiles.py` 都是新实现：前者提供有效能量诊断，后者将已支持的既有数据名称和时间约定接入唯一的公共数据库读取、解码与分析核心。格式兼容不包含或执行历史脚本。本仓库不包含旧源码归档、原仓库历史、生产数据或私人部署配置。另见[发布验证记录](docs/VALIDATION_RESULTS.md)与[依赖及许可边界](THIRD_PARTY.md)。

公开仓库采用 [MIT 许可证](LICENSE)。第三方依赖保留各自许可。代码开源与许可证授权不代表科学结论已经得到认证。

## 既有格式与跨机器适配

新 `legacy_profiles` 模块只负责格式兼容，通过[明确的 writer profile](docs/LEGACY_FORMATS.md)接入公共读取器与分析核心，不另建一套数据流程。机器路径、启动器和环境放在独立配置中，不写成默认值。
