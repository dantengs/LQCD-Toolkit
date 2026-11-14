"""Display contracts tested with explicitly synthetic demonstration values."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lqcd_analysis.plotting import Band, SampleInfo, Series

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class DataContractTests(unittest.TestCase):
    def test_invalid_series(self):
        for kwargs in ({"x": [1j]}, {"y": [np.nan]}, {"x": [[1]]},
                       {"x": [1, 2]}, {"x_unit": ""},
                       {"yerr": [-1], "uncertainty": "sem"},
                       {"yerr": [0.1]}, {"uncertainty": "sem"},
                       {"yerr": [0.1], "uncertainty": "confidence_interval"},
                       {"samples": {"unit": "source"}}):
            values = dict(x=[1], y=[2], label="demonstration", x_unit="a", y_unit="1")
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Series(**values)

    def test_band_contract(self):
        for kwargs in ({"x": [2, 1]}, {"lower": [3, 3]}, {"upper": [1]},
                       {"kind": "confidence"}, {"kind": "prediction", "level": 1.1},
                       {"kind": "mystery"}, {"upper": [2, np.inf]}):
            values = dict(x=[1, 2], lower=[0, 0], upper=[1, 1], label="demonstration",
                          x_unit="a", y_unit="1", kind="spread")
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Band(**values)

    def test_independent_units(self):
        with self.assertRaisesRegex(ValueError, "sources are not independent"):
            SampleInfo("source", 100, "sequential sources")
        with self.assertRaises(ValueError):
            SampleInfo("bin", 0, "blocked configurations")

    def test_input_snapshot_and_asymmetric_errors(self):
        y = np.array([2., 3.])
        s = Series([1, 2], y, "demonstration", "a", "1", [[.1, .2], [.3, .4]], "sem")
        y[0] = 8
        self.assertEqual(s.y[0], 2)
        self.assertFalse(s.y.flags.writeable)

    def test_import_without_matplotlib(self):
        code = '''import sys
sys.modules["matplotlib"] = None
from lqcd_analysis.plotting import Series, plot_series
s = Series([1], [2], "demonstration", "1", "1")
try:
    plot_series([s], xlabel="x", ylabel="y")
except ImportError as exc:
    assert "optional extra" in str(exc)
else:
    raise AssertionError("expected optional dependency error")
'''
        subprocess.run([sys.executable, "-c", code], check=True)


@unittest.skipIf(plt is None, "optional matplotlib is not installed")
class RenderTests(unittest.TestCase):
    def setUp(self):
        self.s = Series([1, 2, 3], [2, 3, 4], "demonstration", "a", "1", [.1]*3, "sem",
                        samples=SampleInfo("demonstration", 3, "synthetic; no scientific inference"))

    def tearDown(self):
        plt.close("all")

    def test_units_and_log_rejected_before_render(self):
        from lqcd_analysis.plotting import plot_series
        bad = Series([1], [1], "other", "fm", "1")
        with self.assertRaisesRegex(ValueError, "units"):
            plot_series([self.s, bad], xlabel="t", ylabel="C")
        for s, kw in ((Series([0], [1], "zero", "a", "1"), {"xscale": "log"}),
                      (Series([1], [.1], "cross zero", "a", "1", [.2], "sd"), {"yscale": "log"})):
            with self.assertRaisesRegex(ValueError, "log"):
                plot_series([s], xlabel="t", ylabel="C", **kw)
        band = Band([1, 2], [-1, 0], [2, 3], "bounds", "a", "1", "spread")
        with self.assertRaisesRegex(ValueError, "log"):
            plot_series([self.s], bands=[band], xlabel="t", ylabel="C", yscale="log")

    def test_exports_provenance_and_style(self):
        from lqcd_analysis.plotting import export_figure, plot_effective_energy
        before = dict(matplotlib.rcParams)
        fig = plot_effective_energy([self.s], title="Demonstration only")
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_figure(fig, Path(tmp)/"demo")
            self.assertEqual([p.suffix for p in paths], [".pdf", ".svg", ".png", ".json"])
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))
            doc = json.loads(paths[-1].read_text(encoding="utf-8"))
            self.assertEqual(doc["series"][0]["samples"]["unit"], "demonstration")
            self.assertEqual(doc["series"][0]["uncertainty"], "sem")
            self.assertEqual(len(doc["series"][0]["y"]["sha256"]), 64)
            self.assertNotIn(tmp, paths[-1].read_text(encoding="utf-8"))
            self.assertIn("<text", paths[1].read_text(encoding="utf-8"))
        self.assertEqual(before, dict(matplotlib.rcParams))

    def test_correlator_fixed_fit_residual(self):
        from lqcd_analysis.plotting import plot_correlator
        fit = Series([1, 2, 3], [1.9, 3.1, 4], "Supplied fit", "a", "1")
        fig = plot_correlator(self.s, fit)
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [.1, -.1, 0], atol=1e-14)
        self.assertIn("not residual-estimator", fig.lqcd_provenance["residual_note"])
        with self.assertRaisesRegex(ValueError, "exactly"):
            plot_correlator(self.s, Series([1, 2], [2, 3], "fit", "a", "1"))

    def test_json_cli(self):
        from lqcd_analysis.plotting import main
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp)/"demo.json"
            spec.write_text(json.dumps({"kind": "form_factors", "title": "Demonstration only",
                "formats": ["png"], "series": [{"x": [0, 1], "y": [1, 2],
                "label": "demonstration", "x_unit": "GeV^2", "y_unit": "1"}]}), encoding="utf-8")
            main([str(spec), str(Path(tmp)/"result")])
            self.assertTrue((Path(tmp)/"result.png").is_file())

    def test_publication_size_and_readable_uncertainty(self):
        from lqcd_analysis.plotting import plot_effective_energy, plot_correlator, export_figure
        size = (86 / 25.4, 70 / 25.4)
        series = Series([1, 2, 3], [2, 3, 4], "Estimate", "a", "1", [.1]*3, "jackknife_se")
        fig = plot_effective_energy([series], figsize_inches=size)
        np.testing.assert_allclose(fig.get_size_inches(), size)
        self.assertEqual(fig.axes[0].get_legend().get_texts()[0].get_text(), "Estimate (jackknife SE)")
        self.assertEqual(fig.lqcd_provenance["series"][0]["uncertainty"], "jackknife_se")
        fit = Series([1, 2, 3], [2, 3, 4], "Fit", "a", "1")
        corr = plot_correlator(series, fit, figsize_inches=size)
        np.testing.assert_allclose(corr.get_size_inches(), size)
        with tempfile.TemporaryDirectory() as tmp:
            # Export records actual dimensions, including an intentional resize.
            fig.set_size_inches(4, 3)
            paths = export_figure(fig, Path(tmp)/"sized", formats=("png",))
            info = json.loads(paths[-1].read_text(encoding="utf-8"))
            self.assertEqual(info["figsize_inches"], [4, 3])
            self.assertEqual(plt.imread(paths[0]).shape[:2], (900, 1200))

    def test_invalid_publication_size(self):
        from lqcd_analysis.plotting import plot_effective_energy
        for size in ((0, 2), (-1, 2), (np.nan, 2), (np.inf, 2), (2,), (2, 3, 4),
                     ((2, 3),), (2j, 3), (True, False)):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "figsize_inches"):
                plot_effective_energy([self.s], figsize_inches=size)
