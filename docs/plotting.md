# Optional scientific plotting

Install with `pip install -e '.[plot]'`. Analysis imports do not load matplotlib;
the plotting module itself imports only NumPy until rendering is requested.
Use `MPLBACKEND=Agg` on a headless machine.

This module **displays existing results**. It does not estimate effective masses,
fit correlators, smooth points, average sources, convert units, calculate z, or
construct uncertainties. It never turns missing data into zeros. Only real,
finite, shape-compatible values are accepted. Project complex quantities in
the analysis layer and name the projection in the label.

## Python API

```python
from lqcd_analysis.plotting import Series, SampleInfo, plot_effective_energy, export_figure

# Demonstration only: invented values, not research evidence.
points = Series(
    x=[3, 4, 5], y=[0.42, 0.40, 0.41], label="Demonstration",
    x_unit="a", y_unit="a^-1", yerr=[0.02, 0.025, 0.03],
    uncertainty="sem",
    samples=SampleInfo("demonstration", 3, "Invented values; no scientific inference"),
)
fig = plot_effective_energy([points], title="Demonstration only")
export_figure(fig, "output/demo")  # PDF, SVG, PNG, provenance JSON
```

Close returned figures with `matplotlib.pyplot.close(fig)` in batch jobs.
All plot functions accept `figsize_inches=(width, height)` (default `(5.2, 3.6)`),
with finite positive dimensions. Set the intended publication size before
export, so a journal layout does not need to shrink the figure and its fonts.
For example, an 86 mm single-column figure with 70 mm height:

```python
# Reuses the synthetic demonstration points above, not scientific evidence.
fig = plot_effective_energy([points], figsize_inches=(86 / 25.4, 70 / 25.4))
fig.axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)
export_figure(fig, "output/single_column_demo")
```

The sidecar records the actual export dimensions in inches, including any
explicit `fig.set_size_inches(...)` resize. Review legends and labels at that
size; an arbitrary amount of text cannot fit a narrow column. Default text is
9 pt and legends are 8 pt at the requested size. The example dimensions are a
caller-selected layout, not a claim of compliance with every journal.

`plot_series(series, xlabel=..., ylabel=..., bands=...)` is the common renderer.
Wrappers choose quantity labels only:

| Function | Supplied data | Output |
| --- | --- | --- |
| `plot_effective_energy` | One or more `Series` | Effective energy; use `ylabel="Effective mass"` for mass |
| `plot_form_factors` | One or more `Series` at q² | Form factors versus q² |
| `plot_z_expansion` | `Series` and explicit `bands=` | Externally evaluated expansion on the supplied x grid; override xlabel for q² |
| `plot_correlator(data, fit, bands=...)` | `Series` data and error-free `Series` fit on exactly the same increasing x grid | Correlator, supplied fit and data-minus-fit residual |

`Series(x, y, label, x_unit, y_unit, yerr=None, uncertainty=None, level=None,
samples=None)` copies arrays. `yerr` contains **deviations**, either symmetric
`(n,)` or `(2,n)` lower/upper deviations. Supported kinds: `sd`, `sem`,
`jackknife_se`, `bootstrap_se`, `confidence_interval`. The last requires
`level=0.68` (or another declared coverage). Error types appear in the legend
as readable labels such as “jackknife SE” and “bootstrap SE”; machine-readable
sidecar enum values are unchanged.
These labels do not prove that the underlying statistical procedure is valid.

`Band(x, lower, upper, label, x_unit, y_unit, kind, level=None, samples=None)`
uses **absolute bounds** on a strictly increasing grid. `kind="confidence"`
means uncertainty of the fitted quantity, `kind="prediction"` means a new
observation's interval, and `kind="spread"` means supplied spread without a
coverage claim. Confidence and prediction require `0 < level < 1`; spread
must not specify a level. Bands are connected piecewise linearly between
supplied bounds, with no smoothing or inferred extension beyond the grid.
No covariance or band is inferred from point error bars. A confidence band is
not a prediction interval; state pointwise versus simultaneous coverage in
the public label/caption. No central curve is invented from band midpoints.

Correlator residual bars repeat **data errors with the fit held fixed**.
They do not estimate uncertainty of a data-minus-fitted-model estimator, since
the fit may be correlated with the data. The legend and sidecar state this.
The fit line joins the supplied predictions without interpolation to extra
points. Dense fit curves on a different grid should be displayed separately
with a supplied `Band`, or prepared as an independent generic figure.

Units are required, use `"1"` for dimensionless. All objects in one panel
must match units exactly; there is no physical dimensional algebra or automatic
conversion. Axis quantity labels are supplied separately from appended units.
`xscale="log"` and `yscale="log"` reject nonpositive points, error endpoints,
or band bounds rather than silently hiding them. Nonfinite points must be
resolved explicitly upstream; this API does not omit them silently.

## Independent samples and provenance

`SampleInfo(unit, count, grouping)` accepts `configuration`, `bin`, or
`demonstration`. The grouping text documents within-configuration source
averaging, bin size and independence justification. Sequential sources,
propagators and noise replicas are **not independent configurations**. Average
sources within each configuration before resampling; consider autocorrelation
and bins. Plotting cannot verify independence. `samples=None` means undeclared
sample information, not inferred n; document it before publication.

For existing toolkit results, obtain `y` and standard errors from the existing
statistics outputs (for example `sqrt(diag(covariance_of_mean))`), preserving
their configuration/bin definition. Evaluate the existing fit model explicitly
at the plotted x values using `FitResult.parameters`; pass those predictions
to the fit `Series`. A plot is not a replacement for fit diagnostics.

`export_figure(fig, stem, formats=("pdf", "svg", "png"), dpi=300)` writes
editable PDF/SVG text and PNG at the declared dpi, plus `stem.provenance.json`.
The sidecar includes canonical float64 array SHA-256 hashes with shape, labels,
units, uncertainty kinds, sample grouping, library versions, style and output
hashes. It records no input/output paths or raw sample identities. It is a
traceability record, **not source data**: retain source arrays separately. Text
fields are copied verbatim, so callers must supply public descriptions without
private paths or identifiers. Replot after changing plotted data so provenance
remains consistent. rcParams are locally scoped; the renderer does not change
the global style. Byte-identical exports across library versions are not
promised (format metadata can include timestamps).

## JSON CLI

Run `lqcd-plot spec.json output/figure` or
`python -m lqcd_analysis.plotting spec.json output/figure`.
The top-level `kind` is `series`, `effective_energy`, `form_factors`,
`z_expansion`, or `correlator`. Arrays are JSON lists. `samples` is an optional
object with the three `SampleInfo` fields. `bands` is a list of `Band` objects.
Correlator requires `data` and `fit`; other kinds require `series`. Remaining
keys are passed to the corresponding Python function. `formats` optionally
selects exports. Unknown fields and invalid data fail with a diagnostic.

```json
{
  "kind": "effective_energy",
  "title": "Demonstration only",
  "series": [{
    "x": [3, 4, 5], "y": [0.42, 0.40, 0.41],
    "label": "Demonstration", "x_unit": "a", "y_unit": "a^-1",
    "yerr": [0.02, 0.025, 0.03], "uncertainty": "sem",
    "samples": {"unit": "demonstration", "count": 3,
                "grouping": "Invented values; no scientific inference"}
  }]
}
```

Run `python tools/plotting_demonstration.py output/demo` for all four plot
types. Every example is explicitly synthetic and must not enter a manuscript
as scientific evidence. Review labels, fit range, uncertainty definitions and
sample grouping before using actual analysis outputs in a publication.
