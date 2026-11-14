"""Optional, display-only scientific plotting; no fitting or resampling.

Contract: quantitative panels display supplied estimates and uncertainties,
not scientific claims inferred by the renderer. Units and uncertainty semantics
are explicit. PDF/SVG keep editable text; PNG is a raster preview. Independence,
autocorrelation and uncertainty coverage remain the caller's responsibility.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.usetex": False, "svg.hashsalt": "lqcd-plot-v1",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
ERROR_KINDS = {"sd", "sem", "jackknife_se", "bootstrap_se", "confidence_interval"}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string (use '1' for dimensionless)")
    return value


def _array(value, name, ndim=1):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
            raise ValueError("not real numeric data")
        a = np.array(raw, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain real numeric data; project complex values explicitly") from exc
    if a.ndim != ndim or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be nonempty, finite and {ndim}-dimensional")
    a.setflags(write=False)
    return a


def _level(value, needed):
    if needed and (isinstance(value, bool) or not isinstance(value, (float, int))
                   or not np.isfinite(value) or not 0 < value < 1):
        raise ValueError("confidence/prediction intervals require 0 < level < 1")
    if not needed and value is not None:
        raise ValueError("level applies only to confidence/prediction intervals")


@dataclass(frozen=True)
class SampleInfo:
    """Declared independent units, never the number of sequential source files.

    'configuration' requires an independence justification; otherwise use bins.
    Synthetic examples use 'demonstration'. No raw IDs or paths are necessary.
    """
    unit: str
    count: int
    grouping: str

    def __post_init__(self):
        if self.unit not in {"configuration", "bin", "demonstration"}:
            raise ValueError("sample unit must be configuration, bin or demonstration; sources are not independent")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("sample count must be a positive integer")
        _text(self.grouping, "sample grouping/independence justification")


@dataclass(frozen=True)
class Series:
    x: object
    y: object
    label: str
    x_unit: str
    y_unit: str
    yerr: object = None
    uncertainty: str = None
    level: float = None
    samples: SampleInfo = None

    def __post_init__(self):
        for name in ("label", "x_unit", "y_unit"):
            _text(getattr(self, name), name)
        x, y = _array(self.x, "x"), _array(self.y, "y")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        if self.samples is not None and not isinstance(self.samples, SampleInfo):
            raise ValueError("samples must be SampleInfo or None")
        if self.yerr is None:
            if self.uncertainty is not None or self.level is not None:
                raise ValueError("uncertainty/level require yerr")
        else:
            raw = np.asarray(self.yerr)
            err = _array(self.yerr, "yerr", ndim=raw.ndim)
            if err.shape not in {(len(x),), (2, len(x))} or np.any(err < 0):
                raise ValueError("yerr must be nonnegative deviations with shape (n,) or (2,n)")
            if self.uncertainty not in ERROR_KINDS:
                raise ValueError(f"uncertainty must be one of {sorted(ERROR_KINDS)}")
            _level(self.level, self.uncertainty == "confidence_interval")
            object.__setattr__(self, "yerr", err)


@dataclass(frozen=True)
class Band:
    x: object
    lower: object
    upper: object
    label: str
    x_unit: str
    y_unit: str
    kind: str
    level: float = None
    samples: SampleInfo = None

    def __post_init__(self):
        for name in ("label", "x_unit", "y_unit"):
            _text(getattr(self, name), name)
        arrays = [_array(getattr(self, name), name) for name in ("x", "lower", "upper")]
        x, low, high = arrays
        if low.shape != x.shape or high.shape != x.shape or np.any(low > high):
            raise ValueError("band bounds must match x shape and lower <= upper")
        if len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("band x must be strictly increasing with at least two points")
        if self.kind not in {"confidence", "prediction", "spread"}:
            raise ValueError("band kind must be confidence, prediction or spread")
        _level(self.level, self.kind in {"confidence", "prediction"})
        if self.samples is not None and not isinstance(self.samples, SampleInfo):
            raise ValueError("samples must be SampleInfo or None")
        for name, a in zip(("x", "lower", "upper"), arrays):
            object.__setattr__(self, name, a)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires the optional extra: pip install 'lqcd-analysis[plot]'") from exc
    return plt


def _hash(a):
    a = np.asarray(a, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode("ascii") + a.tobytes(order="C")).hexdigest()


def _record(obj):
    return {key: {"shape": list(value.shape), "sha256": _hash(value)}
            if isinstance(value, np.ndarray) else value
            for key, value in asdict(obj).items()}


def _validate(series, bands, xscale, yscale):
    if not series or not all(isinstance(s, Series) for s in series):
        raise ValueError("series must contain at least one Series")
    if not all(isinstance(b, Band) for b in bands):
        raise ValueError("bands must contain Band objects")
    if xscale not in {"linear", "log"} or yscale not in {"linear", "log"}:
        raise ValueError("scales must be linear or log")
    units = (series[0].x_unit, series[0].y_unit)
    for obj in (*series, *bands):
        if (obj.x_unit, obj.y_unit) != units:
            raise ValueError("all series/bands must have identical axis units; convert explicitly")
        if xscale == "log" and np.any(obj.x <= 0):
            raise ValueError("log x requires all x values > 0")
        if isinstance(obj, Band):
            low = obj.lower
        else:
            low = obj.y if obj.yerr is None else obj.y - (obj.yerr if obj.yerr.ndim == 1 else obj.yerr[0])
        if yscale == "log" and np.any(low <= 0):
            raise ValueError("log y requires points, error endpoints and band bounds > 0")


def _legend_label(obj):
    kind = obj.kind if isinstance(obj, Band) else obj.uncertainty
    kind = {"jackknife_se": "jackknife SE", "bootstrap_se": "bootstrap SE",
            "confidence_interval": "confidence interval", "sd": "SD", "sem": "SEM"}.get(kind, kind)
    level = f" {100 * obj.level:g}%" if obj.level is not None else ""
    return obj.label + (f" ({kind}{level})" if kind else "")


def _draw(ax, series, bands, xscale, yscale):
    for i, band in enumerate(bands):
        ax.fill_between(band.x, band.lower, band.upper, color=COLORS[i % len(COLORS)],
                        alpha=0.20, label=_legend_label(band))
    for i, s in enumerate(series):
        ax.errorbar(s.x, s.y, yerr=s.yerr, fmt="o", ms=4, capsize=2,
                    color=COLORS[i % len(COLORS)], label=_legend_label(s))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend(fontsize=8)


def plot_series(series, *, xlabel, ylabel, bands=(), title="", xscale="linear", yscale="linear",
                figsize_inches=(5.2, 3.6)):
    """Render supplied points/bounds without interpolation, averaging or fitting.

    Returns a Figure with a provenance snapshot. Axes labels name the quantity;
    units are appended automatically. None sample metadata means undeclared n.
    """
    series, bands = tuple(series), tuple(bands)
    _validate(series, bands, xscale, yscale)
    _text(xlabel, "xlabel")
    _text(ylabel, "ylabel")
    size = _array(figsize_inches, "figsize_inches")
    if size.shape != (2,) or np.any(size <= 0):
        raise ValueError("figsize_inches must contain two finite positive dimensions (width, height)")
    plt = _plt()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=tuple(size), layout="constrained")
        _draw(ax, series, bands, xscale, yscale)
        ax.set(xlabel=f"{xlabel} [{series[0].x_unit}]", ylabel=f"{ylabel} [{series[0].y_unit}]", title=title)
    fig.lqcd_provenance = {"schema": 1, "kind": "series", "xlabel": xlabel,
        "ylabel": ylabel, "title": title, "xscale": xscale, "yscale": yscale,
        "series": [_record(s) for s in series], "bands": [_record(b) for b in bands],
        "style": STYLE.copy(), "figsize_inches": fig.get_size_inches().tolist(),
        "sample_note": "Caller-declared; independence is not verified."}
    return fig


def plot_effective_energy(series, **kwargs):
    """Also suitable for effective mass; pass ylabel='Effective mass'."""
    return plot_series(series, xlabel=kwargs.pop("xlabel", "Time"),
                       ylabel=kwargs.pop("ylabel", "Effective energy"), **kwargs)


def plot_form_factors(series, **kwargs):
    return plot_series(series, xlabel=kwargs.pop("xlabel", r"$q^2$"),
                       ylabel=kwargs.pop("ylabel", "Form factor"), **kwargs)


def plot_z_expansion(series, *, bands, **kwargs):
    """Display externally evaluated z-expansion bounds; does not fit or map q²."""
    return plot_series(series, bands=bands, xlabel=kwargs.pop("xlabel", "z"),
                       ylabel=kwargs.pop("ylabel", "Form factor"), **kwargs)


def plot_correlator(data, fit, *, bands=(), yscale="linear", title="", figsize_inches=(5.2, 3.6)):
    """Fit evaluated at exactly data.x; residual is data minus supplied fit.

    Residual bars repeat data errors (fixed-fit reference), NOT uncertainty of
    the correlated data-minus-fit estimator. No fit parameter errors are added.
    """
    bands = tuple(bands)
    _validate((data, fit), bands, "linear", yscale)
    if not np.array_equal(data.x, fit.x) or fit.yerr is not None:
        raise ValueError("fit must have exactly data.x and no yerr; supply bounds as bands")
    if len(fit.x) > 1 and np.any(np.diff(fit.x) <= 0):
        raise ValueError("fit x must be strictly increasing")
    residual = Series(data.x, data.y-fit.y, "Data minus fixed fit; data errors",
                      data.x_unit, data.y_unit, data.yerr, data.uncertainty, data.level, data.samples)
    fig = plot_series((data,), bands=bands, xlabel="Time", ylabel="Correlator", title=title,
                      yscale=yscale, figsize_inches=figsize_inches)
    plt = _plt()
    with plt.rc_context(STYLE):
        fig.clear()
        ax, res = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        _draw(ax, (data,), bands, "linear", yscale)
        ax.plot(fit.x, fit.y, color=COLORS[1], label=fit.label)
        ax.legend(fontsize=8)
        ax.set(ylabel=f"Correlator [{data.y_unit}]", title=title)
        _draw(res, (residual,), (), "linear", "linear")
        res.axhline(0, color="0.5", linewidth=0.7)
        res.set(xlabel=f"Time [{data.x_unit}]", ylabel=f"Residual [{data.y_unit}]")
    fig.lqcd_provenance.update(kind="correlator", fit=_record(fit), residual=_record(residual),
                               residual_note="Data errors about a fixed-fit reference; not residual-estimator uncertainty.")
    return fig


def export_figure(fig, stem, *, formats=("pdf", "svg", "png"), dpi=300):
    """Save figure and hash-based JSON sidecar; no input/output paths recorded.

    Caller labels/grouping text are copied verbatim: use public descriptions.
    Export snapshots describe original plot inputs; replot after changing data.
    """
    formats = tuple(formats)
    if not formats or any(f not in {"pdf", "svg", "png"} for f in formats):
        raise ValueError("formats must be a nonempty subset of pdf, svg, png")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi must be an integer >= 72")
    if not hasattr(fig, "lqcd_provenance"):
        raise ValueError("figure must be created by lqcd_analysis.plotting")
    import matplotlib
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context(STYLE):
        for fmt in formats:
            path = Path(str(stem) + "." + fmt)
            fig.savefig(path, dpi=dpi, format=fmt)
            paths.append(path)
    info = dict(fig.lqcd_provenance, figsize_inches=fig.get_size_inches().tolist(),
                numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                exports={p.suffix[1:]: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, dpi=dpi)
    sidecar = Path(str(stem) + ".provenance.json")
    sidecar.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return (*paths, sidecar)


def _from_dict(cls, value):
    value = dict(value)
    if value.get("samples") is not None:
        value["samples"] = SampleInfo(**value["samples"])
    return cls(**value)


def main(argv=None):
    """JSON adapter: python -m lqcd_analysis.plotting spec.json output/stem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    fig = None
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        kind = spec.pop("kind", "series")
        formats = spec.pop("formats", ["pdf", "svg", "png"])
        bands = [_from_dict(Band, b) for b in spec.pop("bands", [])]
        if kind == "correlator":
            fig = plot_correlator(_from_dict(Series, spec.pop("data")),
                                  _from_dict(Series, spec.pop("fit")), bands=bands, **spec)
        else:
            functions = {"series": plot_series, "effective_energy": plot_effective_energy,
                         "form_factors": plot_form_factors, "z_expansion": plot_z_expansion}
            if kind not in functions:
                raise ValueError(f"unknown plot kind: {kind}")
            fig = functions[kind]([_from_dict(Series, s) for s in spec.pop("series")], bands=bands, **spec)
        export_figure(fig, args.output, formats=formats)
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        parser.error(str(exc))
    finally:
        if fig is not None:
            _plt().close(fig)


if __name__ == "__main__":
    main()
