# Physics conventions

[中文](PHYSICS_CONVENTIONS.zh-CN.md) · [Overview](../README.md) · [Validation scope](SCIENTIFIC_VALIDATION.md)

This document defines the public `cpu-sequential-reference.v1` profile and the numerical kernels in `lqcd_analysis`. The generation profile is a specific reference contraction convention. It is not a claim that every historical file, solver, meson channel, or current normalization follows that convention.

## 1. Coordinates, array axes, and units

Geometry fields are `nt, nz, ny, nx`. Source positions are supplied as `[x, y, z, t]`, and integer momentum modes as `[nx, ny, nz]`; momentum component names here denote modes rather than lattice extents. In C-order spatial storage, x varies fastest.

| Object | Axes / representation |
|---|---|
| In-memory source or propagator | `(source_spin, source_color, sink_spin, sink_color, t, z, y, x)`, complex128 |
| Raw XQCD-profile file | `(source_spin, source_color, real_imag, sink_spin, sink_color, t, z, y, x)`, big-endian float64, no header |
| `real_imag` | Index 0 is real; index 1 is imaginary |
| Generated small correlator | Time vector in memory; `(2, NT)` real/imaginary array in the generated store |

The raw layout is part of the backend contract. The name of an external executable does not prove that its files use this layout.

Spectral kernels use lattice time and energy units: time is in temporal lattice spacings and E denotes `a_t E`. `lattice_momentum(..., unit="lattice")` returns `a_s p = 2*pi*n/N`; the GeV branch requires a spatial spacing in GeV^-1. Energy and momentum must be converted into the same units before computing q². For anisotropic lattices, the caller supplies the spatial/temporal conversion explicitly. No improved lattice momentum or dispersion relation is chosen automatically.

## 2. Euclidean gamma matrices

`generation.gamma` supports `Id`, `g1` through `g5`, ordered products such as `g5.g1`, and `sigmaij` with distinct i,j in 1…4. The basis has

$$
\gamma_\mu^\dagger=\gamma_\mu,\qquad
\{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu},\qquad
\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4.
$$

Products are evaluated in the written order. In this profile,

$$
\sigma_{ij}=\tfrac12[\gamma_i,\gamma_j]=\gamma_i\gamma_j\quad(i\ne j).
$$

**There is no additional factor of i.** This is a label convention inherited by the reference contraction, not a universal tensor-current definition. Reversing i,j changes the sign; `g5.gi` and `gi.g5` also differ by a sign. Any conversion to another tensor convention must be recorded as a transformation.

## 3. Mass roles and the sequential source

The three masses have ordered roles:

| Role | Symbol | Use |
|---|---|---|
| Initial active | m1 | Initial active propagator and sequential inversion |
| Final active | m2 | Final active propagator, conjugated in the C3 contraction |
| Spectator | m3 | Shared line used to construct the sequential source |

C2 records store masses as `(m3, active_mass)`. C3 records store `(m1,m2,m3)`. Do not sort these tuples. Flavor labels and numerical mass values are user inputs; they do not establish tuning, renormalization, or the identity of a meson.

Let `s=(s_vec,x0)` be the base source and `y0=(x0+T)%NT` the sequential-source slice. For clarity, write propagators in conventional sink-by-source matrix notation, $S_m(u,s)$; the stored source-spin-first array has the corresponding spin axes reversed. The reference source is

$$
\eta(y)=\Gamma_1 S_{m3}(y,s)
\exp[-i\mathbf p_1\cdot(\mathbf y-\mathbf s)],\qquad y_0=(x_0+T)\bmod NT,
$$

and is zero on other time slices. In stored spin indices this means

$$
\eta_{ca}(y)=\sum_b\Gamma_{1,ab}G^{(m3)}_{cb}(y)
\exp[-i\mathbf p_1\cdot(\mathbf y-\mathbf s)].
$$

There is no extra gamma5, dagger, or conjugation at this step. The initial active mass is used for the next inversion. Conditional on the external backend implementing the intended propagator action, the resulting sequential propagator is

$$
Q(x,s)=\sum_{\mathbf y}S_{m1}(x,y)\Gamma_1S_{m3}(y,s)
\exp[-i\mathbf p_1\cdot(\mathbf y-\mathbf s)].
$$

The dense reference backend verifies the equation $D X=\eta$ for the supplied matrix. A production overlap-action output may require additional interpretation; its numerical equivalence to the intended $S_m\eta$ has not been established here.

## 4. Contractions and Fourier phases

For the CPU reference C2, the first propagator is the spectator and is conjugated; the second is the active propagator. In source-first storage, ignoring the displayed color and spacetime sums,

$$
C_2\sim G_{m3}^{*}[n,m]\,G_{\rm active}[l,k]
(\gamma_5\Gamma_{\rm sink})_{mk}
(\Gamma_{\rm source}\gamma_5)_{ln}.
$$

Thus a two-point key written `ga1_ga2` means **sink_source**, not source_sink. Generated two-point sectors currently use equal source and sink operators, but the ordered rule also matters for lower-level off-diagonal contractions.

C3 uses the conjugated ordinary final-active propagator and the un-conjugated sequential propagator:

$$
G_{m2}^{*}[n,m]Q[l,k]
(\gamma_5\Gamma_2)_{mk}(\Gamma_3\gamma_5)_{ln}.
$$

In conventional matrix notation, its closed spin/color expression is

$$
\mathrm{Tr}\!\left[
S_{m2}(x,s)^\dagger\gamma_5\Gamma_2
S_{m1}(x,y)\Gamma_1S_{m3}(y,s)\Gamma_3\gamma_5
\right].
$$

Gamma1 connects m1/m3 at the sequential slice, Gamma2 is the current between m2/m1, and Gamma3 connects m3/m2 at the base source.

The spatial phase function is

$$
\exp\left[-2\pi i\sum_{k=x,y,z}\frac{n_k(r_k-s_k)}{N_k}\right].
$$

C2 uses the selected meson mode. C3 has **two separate phase insertions**: the initial-mode phase already included in the sequential source at y, and the mode `momentum_initial - momentum_final` applied at the current position x during contraction. Do not apply the initial phase a second time after inversion. These definitions specify this reference profile; a differently generated sequential propagator requires a separately verified profile.

## 5. Stored time versus physical fit time

Sources and propagators retain absolute lattice coordinates until contraction output is reordered:

$$
C_2[j]=C_{2,\rm raw}[(x_0+j)\bmod NT],
$$

$$
C_3[j]=C_{3,\rm raw}[(x_0+T-j)\bmod NT].
$$

For this sequential convention, j=0 is the initial insertion at x0+T and j=T is the final insertion at x0. Increasing j runs backward in raw lattice time. Generated C3 selections label this physical initial-to-current coordinate as `initial-forward`, with the raw mapping retained in metadata. Known historical `sink-backward` data can instead use the explicit `cpu-st` analysis mapping.

Do not shift an already reordered array again. Do not derive time orientation from shape alone. The source generator performs the stated modulo selection and does not add an independent boundary-crossing sign; the external action's temporal boundary condition remains an explicit backend input.

## 6. Point and Z3 sources

Point sources have unit amplitude and twelve spin/color-diluted right-hand sides. Z3-grid sources use reproducible NumPy PCG64 draws of cube roots of unity on a periodic spatial support at one time, with the same spatial noise across the diluted right-hand sides. The same source is used for the relevant mass solves and subsequent linear sequential construction. The stream is not asserted to reproduce an external solver's RNG.

For independent noise phases,

$$
\mathbb E[\theta_r^*\theta_{r'}]=\delta_{rr'}.
$$

The quadratic correlator estimator therefore reduces to a sum over supported source points in noise expectation. Dividing C2 and C3 by the **support-site count** gives a source-average estimator. Cross-source terms need not vanish for an individual realization.

The current profile only accepts this estimator for zero initial and final momentum, with the explicit choice `zero-momentum-source-average`. It does not generalize this normalization to arbitrary momentum-projected grid sources. The source/time identity currently supports one measurement per configuration and source time; additional independent noise replicas require a richer identity contract, not a renamed duplicate.

## 7. Spectral models and statistical assumptions

Define $Z_n=\langle0|O|n\rangle$, with the explicit relativistic normalization factors used below. The diagonal periodic model is

$$
C_2(t)=\sum_n\frac{|Z_n|^2}{2E_n}
\left[e^{-E_nt}+e^{-E_n(NT-t)}\right].
$$

It assumes the same source/sink operator and equal forward/backward amplitudes. The forward contribution for $0\le\tau\le T\le NT$ is

$$
C_3(\tau,T)=\sum_{m,n}
\frac{Z_{f,m}Z_{i,n}^{*}M_{mn}}{4E_{f,m}E_{i,n}}
e^{-E_{f,m}(T-\tau)-E_{i,n}\tau}.
$$

`matrix_elements` has axes `(final_state, initial_state)`. This kernel is not a full finite-temperature correlator: it omits thermal transitions, wrapping contributions, oscillating terms, and contact terms. The general kernel allows endpoint evaluation algebraically; the joint fitting interface excludes C3 contact endpoints 0 and T.

The joint model uses positive ordered energy gaps, positive real overlaps, and real matrix elements after an explicit current projection. Given complex C3 data, the fitted observable is `scale * real(phase * C3)`, with a declared unit complex phase and evidence. Choosing the visually stronger real/imaginary part is not a convention determination.

Sources are averaged within each configuration before configuration alignment, binning, and resampling. The joint covariance retains cross-channel correlations. Equal-size bins and their membership are explicit; data ordering, chain boundaries, independence, and a sufficient bin size remain the caller's responsibility. Fixed covariance SVD cuts and fixed windows are used for the jackknife fits. The joint model supplies analytic derivatives so finite-difference noise cannot falsely resolve coincident transition exponents. Parameter rank is checked for the central fit and all retained jackknife samples; failed samples are not silently removed.

## 8. P→P tools and z parameterization

`pp_form_factors` accepts an **already renormalized real contravariant Minkowski** vector current `(V0,Vx,Vy,Vz)`, with metric (+---), and continuum on-shell kinematics in one declared unit. It solves

$$
V^\mu=(p_i+p_f)^\mu f_+ +(p_i-p_f)^\mu f_-,\qquad
f_0=f_+ +\frac{q^2}{M_i^2-M_f^2}f_-.
$$

This avoids division by q² and implements $f_0(0)=f_+(0)$. It rejects equal/near-equal masses, rank-deficient or ill-conditioned kinematics such as zero recoil for separate f± extraction, and incompatible current components. It is an algebraic consistency/extraction tool, not a covariance-weighted fit to noisy currents or an automatic finite-lattice-spacing correction. The standard P→P definition is given in [the original P→P parameterization paper, Eq. (1)](https://arxiv.org/pdf/0807.2722).

`z_map` uses explicit tplus and t0:

$$
z(q^2)=\frac{\sqrt{t_+-q^2}-\sqrt{t_+-t_0}}
{\sqrt{t_+-q^2}+\sqrt{t_+-t_0}}.
$$

The real implementation requires q²<tplus and t0<tplus. `pole_z_series` evaluates an ascending polynomial in z divided by `1-q²/pole_mass²`; `pole_mass=None` explicitly selects no pole. It is **not automatically BCL or BGL**: no threshold derivative relation, outer function, or unitarity bound is supplied. The original BCL construction includes additional physical structure; see [Secs. II–III of the original paper](https://arxiv.org/pdf/0807.2722). Two independent series do not automatically share the f0(0)=f+(0) constraint. Tensor form factors must not be tied to that condition without an additional justified assumption.

The branch point and pole belong to the current/channel analysis and must be supplied explicitly. A vector-meson pole in a P→P parameterization is not evidence that the initial state is a vector meson.

P→V and V→P need explicit polarization, vector/axial decomposition, phase, and normalization conventions. For an example of the additional structures in P→V, see [the P→V lattice study, Eqs. (1)–(3)](https://arxiv.org/pdf/1311.6669). The present package does not automatically extract these form factors, even when it can generate the corresponding gamma-labeled correlators.
