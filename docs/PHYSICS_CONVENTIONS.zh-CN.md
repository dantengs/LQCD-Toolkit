# 物理约定

[English](PHYSICS_CONVENTIONS.md) · [项目介绍](../README.zh-CN.md) · [验证范围](SCIENTIFIC_VALIDATION.md)

本文定义公开生成 profile `cpu-sequential-reference.v1` 与 `lqcd_analysis` 数值内核的约定。它描述一套明确的参考收缩，不代表所有历史文件、外部求解器、介子通道或流归一化都自动满足同一约定。

## 1. 坐标、数组轴与单位

格点尺寸使用 `nt, nz, ny, nx`。源位置按 `[x,y,z,t]` 输入，整数动量模式按 `[nx,ny,nz]` 输入；这里动量中的 n 表示模式数。空间 C-order 存储以 x 为最快变化的指标。

| 对象 | 轴或表示 |
|---|---|
| 内存中的源/传播子 | `(source_spin, source_color, sink_spin, sink_color, t, z, y, x)`，complex128 |
| XQCD-profile 原始文件 | `(source_spin, source_color, real_imag, sink_spin, sink_color, t, z, y, x)`，无头 big-endian float64 |
| `real_imag` | 0为实部，1为虚部 |
| 自产小关联函数 | 内存中为时间向量；入库后为 `(2,NT)` 实/虚数组 |

原始布局必须是后端契约的一部分；可执行程序的名称不能证明文件采用这种布局。

谱模型使用格点时间和能量：时间以时间格距为单位，E表示 `a_t E`。`lattice_momentum(..., unit="lattice")` 返回 `a_s p = 2*pi*n/N`；GeV分支要求显式输入单位为GeV^-1的空间格距。计算q²前，能量和动量必须换到相同单位。各向异性格点的时间/空间格距换算由调用方明确提供；程序不自动选择改进格点动量或色散关系。

## 2. Euclidean gamma 矩阵

`generation.gamma` 支持 `Id`、`g1` 至 `g5`、如 `g5.g1` 的有序乘积，以及1…4中不同指标构成的 `sigmaij`。所用基满足

$$
\gamma_\mu^\dagger=\gamma_\mu,\quad
\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu},\quad
\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4.
$$

乘积按书写顺序计算。本profile的张量标签为

$$
\sigma_{ij}=\tfrac12[\gamma_i,\gamma_j]=\gamma_i\gamma_j\quad(i\ne j).
$$

**这里没有额外的 i。** 这是参考收缩继承的标签定义，不是普适的张量流定义。交换i,j会变号，`g5.gi`与`gi.g5`也相差负号。转换到其他张量约定时必须记录变换。

## 3. 质量角色与顺序源

三条质量线有明确顺序：

| 角色 | 符号 | 用途 |
|---|---|---|
| 初态 active | m1 | 初态 active 普通传播子，以及顺序源反演 |
| 末态 active | m2 | 末态 active 普通传播子，在C3中取共轭 |
| Spectator | m3 | 共享线，用于构造顺序源 |

C2记录采用 `(m3, active_mass)`，C3采用 `(m1,m2,m3)`；不能对质量元组排序。Flavor标签和质量数值来自用户配置，不自动完成质量调谐、重整化或介子身份鉴定。

令基础源为 `s=(s_vec,x0)`，顺序源切片为 `y0=(x0+T)%NT`。为便于书写，用通常的sink×source矩阵记号 $S_m(u,s)$；原始数组将source spin放在前面，因此这两个spin轴的顺序相反。参考顺序源为

$$
\eta(y)=\Gamma_1S_{m3}(y,s)
\exp[-i\mathbf p_1\cdot(\mathbf y-\mathbf s)],\qquad y_0=(x_0+T)\bmod NT,
$$

其他时间切片为零。对应原始存储指标：

$$
\eta_{ca}(y)=\sum_b\Gamma_{1,ab}G^{(m3)}_{cb}(y)
\exp[-i\mathbf p_1\cdot(\mathbf y-\mathbf s)].
$$

此处不再加gamma5、dagger或复共轭。下一次反演采用初态active质量m1。如果外部后端确实实现所需传播子作用，则

$$
Q(x,s)=\sum_{\mathbf y}S_{m1}(x,y)\Gamma_1S_{m3}(y,s)
\exp[-i\mathbf p_1\cdot(\mathbf y-\mathbf s)].
$$

稠密参考后端对用户提供的矩阵验证 $DX=\eta$。真实overlap后端的输出可能有额外的传播子定义或处理；本项目尚未对其与目标 $S_m\eta$ 的数值等价性作验证。

## 4. 收缩与 Fourier 相位

CPU参考C2的第一传播子为spectator并取共轭，第二传播子为active。省略显式颜色和时空求和后，source-first指标表达为

$$
C_2\sim G_{m3}^{*}[n,m]G_{\rm active}[l,k]
(\gamma_5\Gamma_{\rm sink})_{mk}
(\Gamma_{\rm source}\gamma_5)_{ln}.
$$

因此二点函数键 `ga1_ga2` 的端点顺序是 **sink_source**。目前生成的C2扇区使用相同源汇算符；底层非对角收缩仍需保持这个顺序。

C3取普通末态active传播子的复共轭，与未共轭的顺序传播子收缩：

$$
G_{m2}^{*}[n,m]Q[l,k]
(\gamma_5\Gamma_2)_{mk}(\Gamma_3\gamma_5)_{ln}.
$$

通常矩阵记号下的spin/color闭合式为

$$
\mathrm{Tr}\!\left[
S_{m2}(x,s)^\dagger\gamma_5\Gamma_2
S_{m1}(x,y)\Gamma_1S_{m3}(y,s)\Gamma_3\gamma_5
\right].
$$

Gamma1在顺序源切片连接m1/m3；Gamma2为连接m2/m1的流；Gamma3在基础源位置连接m3/m2。

空间相位函数为

$$
\exp\left[-2\pi i\sum_{k=x,y,z}\frac{n_k(r_k-s_k)}{N_k}\right].
$$

C2使用指定介子动量模式。C3有**位置不同的两次相位插入**：顺序源在y处已经包含初态动量相位；收缩时在current位置x处另乘模式 `momentum_initial - momentum_final` 的相位。反演后不能给顺序传播子重复乘初态相位。其他方式生成的顺序传播子必须使用另外验证的profile。

## 5. 存储时间与拟合的物理时间

源和传播子保持绝对格点坐标，直到收缩输出阶段才重排：

$$
C_2[j]=C_{2,\rm raw}[(x_0+j)\bmod NT],
$$

$$
C_3[j]=C_{3,\rm raw}[(x_0+T-j)\bmod NT].
$$

在该顺序源约定下，j=0对应x0+T的初态插入，j=T对应x0的末态插入。j增大时原始格点时间递减。自产C3清单用 `initial-forward` 标记这个从初态到current的物理坐标，并保留原始映射。已确认采用历史 `sink-backward` 存储的数据，可显式选择 `cpu-st` 分析映射。

已重排数组不能再重复源平移，也不能仅凭形状推断时间方向。源生成器只做上述modulo切片选择，不独立添加跨时间边界符号；外部action的时间边界条件仍是明确的后端输入。

## 6. 点源与 Z3 网格源

点源为单位强度，带十二个自旋/颜色稀释右端项。Z3网格源在一个时间片的周期空间支撑上，用NumPy PCG64可复现地产生三次单位根，并在稀释右端项之间共享空间噪声。有关质量的传播子和后续线性顺序源构造使用同一份源。这里不声称复现其他求解器的随机数流。

独立噪声满足

$$
\mathbb E[\theta_r^*\theta_{r'}]=\delta_{rr'}.
$$

因此二次关联函数估计量在噪声期望下变为支撑点上的求和。C2和C3除以**非零支撑点数**后，得到源平均的估计量。单次噪声实现中的跨源交叉项不一定为零。

当前profile仅在初末态动量均为零、且显式声明 `zero-momentum-source-average` 时接受该估计量，不将归一化推广到任意非零动量网格源。当前测量身份只支持每组态每源时间一个测量；增加独立噪声副本需要扩展身份契约，不能只给重复测量换名称。

## 7. 谱模型与统计假设

定义 $Z_n=\langle0|O|n\rangle$，并采用下面明确写出的相对论归一化因子。对角周期模型为

$$
C_2(t)=\sum_n\frac{|Z_n|^2}{2E_n}
\left[e^{-E_nt}+e^{-E_n(NT-t)}\right].
$$

它假设源汇为相同算符，前后向振幅相等。在 $0\le\tau\le T\le NT$ 中，前向三点贡献为

$$
C_3(\tau,T)=\sum_{m,n}
\frac{Z_{f,m}Z_{i,n}^{*}M_{mn}}{4E_{f,m}E_{i,n}}
e^{-E_{f,m}(T-\tau)-E_{i,n}\tau}.
$$

`matrix_elements` 轴顺序为 `(final_state, initial_state)`。这不是完整有限温关联函数：未包括热跃迁、绕行、振荡项或接触项。底层数值内核可代数计算端点，联合拟合接口则排除C3的0和T接触端点。

联合模型使用正的有序能隙、正实overlap和显式投影后的实矩阵元。对复C3，拟合量为 `scale * real(phase * C3)`；必须指定单位复相位并给出依据。不能以实部/虚部哪一个信号更强来代替约定推导。

先在每个组态内平均源，再对齐组态、分箱和重采样。联合协方差保留跨通道相关。等大小bins及成员明确记录；数据顺序、多链边界、独立性和分箱长度是否充分，仍由调用方判断。Jackknife拟合使用固定窗口与固定的协方差SVD截断。联合模型提供解析导数，避免有限差分噪声将重合跃迁指数误判为独立信息；中心拟合与所有jackknife样本均检查参数秩，失败样本不被静默删除。

## 8. P→P 工具与 z 参数化

`pp_form_factors` 接受**已经重整化的、实的、contravariant Minkowski**矢量流 `(V0,Vx,Vy,Vz)`，度规为(+---)，输入必须是同一单位下的连续体on-shell运动学。它求解

$$
V^\mu=(p_i+p_f)^\mu f_+ +(p_i-p_f)^\mu f_-,\qquad
f_0=f_+ +\frac{q^2}{M_i^2-M_f^2}f_-.
$$

该形式不除以q²，并满足 $f_0(0)=f_+(0)$。等/近等质量、秩亏或病态运动学（例如单独提取f±时的零反冲）以及不满足分解的流分量都会被拒绝。这是代数一致性与提取工具，不是对带噪声流数据的协方差拟合，也不自动修正有限格距效应。P→P标准定义见 [P→P参数化原始论文的式(1)](https://arxiv.org/pdf/0807.2722)。

`z_map` 要求显式提供tplus和t0：

$$
z(q^2)=\frac{\sqrt{t_+-q^2}-\sqrt{t_+-t_0}}
{\sqrt{t_+-q^2}+\sqrt{t_+-t_0}}.
$$

实轴实现要求q²<tplus、t0<tplus。`pole_z_series` 按z升幂系数求多项式，再除以 `1-q²/pole_mass²`；`pole_mass=None`明确表示无极点。它**不自动等于BCL或BGL**：未提供阈值导数关系、outer function或unitarity bound。BCL的额外物理结构见[原始论文第II–III节](https://arxiv.org/pdf/0807.2722)。两条独立级数不会自动满足f0(0)=f+(0)，更不能无依据将张量形状因子也绑定到这个约束。

分支点和极点属于具体流与通道的分析，需要显式指定。P→P参数化中出现向量介子极点，并不说明初态是向量介子。

P→V与V→P还需要明确极化、vector/axial分解、相位和归一化。P→V的额外结构示例可见 [P→V格点研究的式(1)–(3)](https://arxiv.org/pdf/1311.6669)。即使可以生成对应gamma标签的关联函数，本包也不自动提取这些形状因子。
