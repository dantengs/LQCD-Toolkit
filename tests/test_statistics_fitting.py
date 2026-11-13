"""Analytic/synthetic checks; never import legacy scripts or access research data."""
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from lqcd_analysis.statistics import (
    ConfigSamples, average_sources, align_channels, bin_configurations,
    mean_covariance, jackknife, jackknife_covariance, bootstrap,
)
from lqcd_analysis.fitting import correlated_fit, fit_jackknife, FitError, JackknifeFitError


UNBOUNDED = (-np.inf, np.inf)


def line(x, p):
    return p[0] + p[1] * x


class StatisticsTests(unittest.TestCase):
    def test_source_average_before_equal_config_weight(self):
        samples = average_sources(["a", "a", "b"], [0, 2, 9], [0, 1, 0])
        self.assertEqual(samples.config_ids, ("a", "b"))
        assert_allclose(samples.values, [1, 9])
        self.assertEqual(float(jackknife(samples).mean), 5)
        self.assertNotEqual(float(samples.values.mean()), np.mean([0, 2, 9]))
        assert_allclose(mean_covariance(samples), [[16]])

    def test_duplicate_and_nonfinite_data_rejected(self):
        with self.assertRaises(ValueError):
            average_sources(["a", "a"], [1, 2], [0, 0])
        for values in ([1, np.nan], [1, np.inf], [1 + 0j, 2 + 0j]):
            with self.assertRaises(ValueError):
                ConfigSamples(("a", "b"), values)
        with self.assertRaises(ValueError):
            ConfigSamples(("a", "a"), [1, 2])

    def test_exact_alignment_uses_identity_and_intersection_reports_drops(self):
        a = ConfigSamples(("a", "b", "c"), [1, 2, 3])
        b = ConfigSamples(("c", "a", "b"), [30, 10, 20])
        aligned = align_channels({"a": a, "b": b})
        assert_array_equal(aligned.channels["b"].values, [10, 20, 30])
        c = ConfigSamples(("b", "c", "d"), [20, 30, 40])
        with self.assertRaises(ValueError):
            align_channels({"a": a, "c": c})
        intersected = align_channels({"a": a, "c": c}, "intersection")
        self.assertEqual(intersected.channels["a"].config_ids, ("b", "c"))
        self.assertEqual(intersected.dropped_ids, {"a": ("a",), "c": ("d",)})

    def test_ordered_binning_and_remainder_fail(self):
        samples = ConfigSamples((9, 3, 7, 1), [1, 3, 10, 14])
        binned = bin_configurations(samples, 2)
        self.assertEqual(binned.config_ids, ((9, 3), (7, 1)))
        assert_allclose(binned.values, [2, 12])
        assert_allclose(jackknife(binned).covariance, [[25]])
        with self.assertRaises(ValueError):
            bin_configurations(samples, 3)
        with self.assertRaises(ValueError):
            bin_configurations(samples, True)

    def test_jackknife_covariance_matches_analytic_mean_covariance(self):
        samples = ConfigSamples(("a", "b", "c"), [[1, 2], [2, 4], [3, 6]])
        result = jackknife(samples)
        assert_allclose(result.samples, [[2.5, 5], [2, 4], [1.5, 3]])
        assert_allclose(result.covariance, [[1/3, 2/3], [2/3, 4/3]])
        assert_allclose(result.covariance, mean_covariance(samples))
        # Original covariance is three times the covariance of the mean.
        assert_allclose(np.cov(samples.values, rowvar=False), 3 * result.covariance)

    def test_paired_bootstrap_is_reproducible_and_preserves_cross_covariance(self):
        channels = {"a": ConfigSamples((1, 2, 3), [1, 2, 4]),
                    "b": ConfigSamples((3, 1, 2), [8, 2, 4])}
        first = bootstrap(channels, 200, seed=19)
        second = bootstrap(channels, 200, seed=19)
        assert_array_equal(first.indices, second.indices)
        assert_array_equal(first.samples["a"], second.samples["a"])
        assert_allclose(first.samples["b"], 2 * first.samples["a"])
        assert_allclose(first.covariance[0, 1], 2 * first.covariance[0, 0])
        self.assertEqual(first.channel_slices, {"a": slice(0, 1), "b": slice(1, 2)})

    def test_insufficient_independent_samples_fail(self):
        samples = ConfigSamples((1,), [7])
        for fn in (jackknife, mean_covariance):
            with self.assertRaises(ValueError):
                fn(samples)
        with self.assertRaises(ValueError):
            bootstrap({"x": samples}, 20, seed=0)


class FittingTests(unittest.TestCase):
    def test_correlated_fit_agrees_with_gls_normal_equations(self):
        x = np.arange(4.)
        y = np.array([1.2, 2.9, 5.3, 6.8])
        cov = np.array([[2., .8, 0, 0], [.8, 1., .2, 0], [0, .2, 3., .5], [0, 0, .5, 2.]]) / 100
        design = np.column_stack((np.ones(4), x))
        expected_cov = np.linalg.inv(design.T @ np.linalg.solve(cov, design))
        expected = expected_cov @ design.T @ np.linalg.solve(cov, y)
        fit = correlated_fit(x, y, cov, line, [0., 0.], bounds=UNBOUNDED)
        assert_allclose(fit.parameters, expected, rtol=1e-6)
        assert_allclose(fit.parameter_covariance, expected_cov, rtol=1e-6)
        residual = y - design @ expected
        assert_allclose(fit.chi2, residual @ np.linalg.solve(cov, residual), rtol=1e-10)
        self.assertEqual((fit.rank, fit.parameter_rank, fit.dof), (4, 2, 2))
        self.assertIsNotNone(fit.pvalue)

    def test_covariance_scaling_changes_uncertainty_and_chi2_correctly(self):
        x, y = np.arange(4.), [1., 3., 6., 6.]
        a = correlated_fit(x, y, np.eye(4), line, [0., 0.], bounds=UNBOUNDED)
        b = correlated_fit(x, y, np.eye(4) / 10, line, [0., 0.], bounds=UNBOUNDED)
        assert_allclose(a.parameters, b.parameters, rtol=1e-6)
        assert_allclose(a.parameter_covariance / 10, b.parameter_covariance, rtol=1e-6)
        assert_allclose(a.chi2 * 10, b.chi2)

    def test_singular_covariance_uses_retained_rank(self):
        cov = np.array([[1., 0, 1], [0, 1, 0], [1, 0, 1]])
        fit = correlated_fit(np.arange(3.), [2., 3., 2.], cov,
                             lambda x, p: np.full(x.shape, p[0]), [0.], bounds=UNBOUNDED)
        self.assertEqual((fit.rank, fit.parameter_rank, fit.dof), (2, 1, 1))
        assert_allclose(fit.parameters, [2.5], atol=1e-7)
        assert_allclose(fit.chi2, .5)
        assert_allclose(fit.parameter_covariance, [[.5]], rtol=1e-6)

    def test_cutoff_controls_effective_dof(self):
        model = lambda x, p: np.full(x.shape, p[0])
        fit = correlated_fit(np.arange(3.), [1., 1., 1.], np.diag([1., .1, 1e-14]),
                             model, [0.], bounds=UNBOUNDED, svd_rcond=1e-10)
        self.assertEqual((fit.rank, fit.dof), (2, 1))

    def test_zero_dof_and_active_bounds_omit_pvalue(self):
        model = lambda x, p: np.full(x.shape, p[0])
        zero = correlated_fit([0.], [1.], [[1.]], model, [0.], bounds=UNBOUNDED)
        self.assertEqual(zero.dof, 0)
        self.assertIsNone(zero.pvalue)
        bounded = correlated_fit([0., 1., 2.], [-2., -2., -2.], np.eye(3), model,
                                 [1.], bounds=([0.], [np.inf]))
        self.assertIsNone(bounded.pvalue)
        self.assertIsNone(bounded.parameter_covariance)

    def test_unidentifiable_parameter_is_reported(self):
        fit = correlated_fit([0., 1., 2.], [2., 2., 2.], np.eye(3),
                             lambda x, p: np.full(x.shape, p[0] + p[1]),
                             [0., 0.], bounds=UNBOUNDED)
        self.assertEqual(fit.parameter_rank, 1)
        self.assertIsNone(fit.parameter_covariance)
        self.assertIsNone(fit.pvalue)

    def test_invalid_covariance_and_nonfinite_model_fail(self):
        for cov in (np.zeros((3, 3)), np.diag([1., -1., 1.]),
                    np.array([[1., .2, 0], [0, 1., 0], [0, 0, 1.]]),
                    np.diag([1., np.inf, 1.])):
            with self.assertRaises(ValueError):
                correlated_fit(np.arange(3.), np.ones(3), cov, line, [0., 0.], bounds=UNBOUNDED)
        with self.assertRaises(FitError):
            correlated_fit(np.arange(3.), np.ones(3), np.eye(3),
                           lambda x, p: x * np.nan, [0.], bounds=UNBOUNDED)
        with self.assertRaises(FitError):
            correlated_fit(np.arange(3.), [1., 3., 5.], np.eye(3), line,
                           [0., 0.], bounds=UNBOUNDED, max_nfev=1)

    def test_jackknife_uses_same_full_covariance_and_window(self):
        x = np.arange(4.)
        raw = 1 + 2 * x + np.random.default_rng(5).normal(0, .1, size=(12, 4))
        data = jackknife(ConfigSamples(tuple(range(12)), raw))
        with patch("lqcd_analysis.fitting.correlated_fit", wraps=correlated_fit) as calls:
            fit = fit_jackknife(x, data, line, [0., 0.], window=[1, 2, 3], bounds=UNBOUNDED)
        self.assertEqual(len(calls.call_args_list), 13)
        for call in calls.call_args_list:
            assert_array_equal(call.args[0], [1., 2., 3.])
            assert_array_equal(call.args[2], data.covariance[1:, 1:])
        assert_allclose(fit.parameter_covariance, fit.full_fit.parameter_covariance, rtol=2e-5)
        assert_allclose(fit.bias_corrected_parameters, fit.full_fit.parameters, atol=1e-6)

    def test_jackknife_failure_reports_identity_and_does_not_drop_sample(self):
        raw = np.random.default_rng(2).normal(size=(6, 3))
        data = jackknife(ConfigSamples(tuple("abcdef"), raw))
        full = correlated_fit(np.arange(3.), data.mean, data.covariance,
                              line, [0., 0.], bounds=UNBOUNDED)
        with patch("lqcd_analysis.fitting.correlated_fit", side_effect=[full, FitError("injected nonconvergence")]):
            with self.assertRaises(JackknifeFitError) as caught:
                fit_jackknife(np.arange(3.), data, line, [0., 0.], window=slice(None), bounds=UNBOUNDED)
        self.assertEqual(caught.exception.sample_index, 0)
        self.assertEqual(caught.exception.config_id, "a")


class LegacyFormulaReproductionTests(unittest.TestCase):
    """Independent arithmetic reproductions, not legacy script execution."""

    def test_including_full_mean_as_replicate_changes_jackknife_variance(self):
        valid = jackknife_covariance([1., 2., 3.])[0, 0]
        contaminated = jackknife_covariance([2., 1., 2., 3.])[0, 0]
        assert_allclose(valid, 4/3)
        assert_allclose(contaminated, 1.5)

    def test_completion_order_can_replace_the_full_estimate(self):
        def old_bias_corrected(values):
            n = len(values) - 1
            return np.mean(n * values[0] - (n - 1) * np.array(values[1:]))
        self.assertEqual(old_bias_corrected([10., 8., 12.]), 10.)
        self.assertEqual(old_bias_corrected([8., 10., 12.]), 5.)

    def test_diagonal_residual_is_not_correlated_chi_square(self):
        r, cov = np.ones(2), np.array([[1., .9], [.9, 1.]])
        self.assertEqual(float(r @ r), 2.)
        assert_allclose(r @ np.linalg.solve(cov, r), 2 / 1.9)

    def test_parameter_dependent_residual_scale_needs_derivative(self):
        # y=3, x=1, sx=sy=1, f=p*x: r=(3-p)/sqrt(1+p^2).
        residual = lambda p: (3 - p) / np.sqrt(1 + p*p)
        h = 1e-5
        derivative = (residual(1 + h) - residual(1 - h)) / (2*h)
        assert_allclose(derivative, -np.sqrt(2), rtol=1e-8)
        self.assertNotAlmostEqual(derivative, -1 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
