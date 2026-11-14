"""Synthetic display demonstration only; never manuscript evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lqcd_analysis.plotting import (
    Series, Band, SampleInfo, plot_effective_energy, plot_correlator,
    plot_form_factors, plot_z_expansion, export_figure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    samples = SampleInfo("demonstration", 3, "Invented values; no scientific inference")
    title = "Demonstration only — synthetic values"
    t = np.arange(3, 9)
    energy = Series(t, [.42, .40, .41, .405, .399, .403], "Demonstration", "a", "a^-1",
                    [.015, .02, .02, .025, .03, .035], "sem", samples=samples)
    fig = plot_effective_energy([energy], title=title)
    export_figure(fig, args.output/"effective_energy")
    plt.close(fig)
    y = np.exp(-.4*t)
    data = Series(t, y*np.array([1.03, .98, 1.01, .97, 1.02, .99]), "Demonstration",
                  "a", "1", y*.04, "sem", samples=samples)
    fit = Series(t, y, "Supplied illustrative curve", "a", "1")
    fig = plot_correlator(data, fit, title=title, yscale="log")
    export_figure(fig, args.output/"correlator")
    plt.close(fig)
    q = np.array([0., .4, .8, 1.2])
    ff = Series(q, [.7, .75, .83, .95], "Demonstration", "GeV^2", "1", [.03]*4,
                "sem", samples=samples)
    fig = plot_form_factors([ff], title=title)
    export_figure(fig, args.output/"form_factors")
    plt.close(fig)
    z = np.linspace(-.1, .1, 30)
    center = .8-.6*z
    band = Band(z, center-.04, center+.04, "Illustrative bounds", "1", "1",
                "confidence", .68, samples)
    zp = Series([-.08, 0, .08], [.85, .79, .76], "Demonstration", "1", "1", [.03]*3,
                "sem", samples=samples)
    fig = plot_z_expansion([zp], bands=[band], title=title)
    export_figure(fig, args.output/"z_expansion")
    plt.close(fig)


if __name__ == "__main__":
    main()
