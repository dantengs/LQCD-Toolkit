"""Analytic checks and tiny independent reconstructions; no legacy imports."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from lqcd_analysis.physics import (
    c2_periodic, c3_forward, lattice_momentum, q_squared,
    pp_form_factors, z_map, pole_z_series,
)


class SpectralTests(unittest.TestCase):
    def test_c2_normalization_periodicity_and_complex_overlap(self):
        t = np.array([0, 3, 8, 16])
        actual = c2_periodic(t, 16, [0.4], [3+4j])
        expected = 25/0.8*(np.exp(-0.4*t)+np.exp(-0.4*(16-t)))
        assert_allclose(actual, expected)
        assert_allclose(actual, c2_periodic(16-t, 16, [0.4], [3+4j]))

    def test_c2_multiple_states(self):
        expected = sum(abs(z)**2/(2*e)*(np.exp(-2*e)+np.exp(-8*e))
                       for e, z in zip([0.3, 0.9], [2, 0.4]))
        self.assertAlmostEqual(c2_periodic(2, 10, [0.3, 0.9], [2, 0.4]), expected)

    def test_c3_independent_double_sum_with_complex_phases(self):
        ei, ef = [0.4, 0.7], [0.2, 0.9, 1.2]
        zi, zf = [1+2j, 0.3], [2-1j, 0.7, 0.2j]
        matrix = np.array([[1+0.1j, 2], [3, 4j], [5, 6]])
        tau = np.array([0, 1, 4, 7])
        expected = []
        for t in tau:
            total = 0j
            for m in range(3):
                for n in range(2):
                    total += zf[m]*np.conj(zi[n])*matrix[m,n]/(4*ef[m]*ei[n])*np.exp(-ef[m]*(7-t)-ei[n]*t)
            expected.append(total)
        assert_allclose(c3_forward(tau, 7, 32, ei, ef, zi, zf, matrix), expected)

    def test_c3_has_explicit_source_to_sink_orientation(self):
        tau = np.arange(7)
        value = c3_forward(tau, 6, 32, [0.8], [0.3], [2], [3], [[4]])
        assert_allclose(value[1:]/value[:-1], np.exp(-0.5))
        assert_allclose(value, c3_forward(tau, 6, 64, [0.8], [0.3], [2], [3], [[4]]))

    def test_invalid_spectral_inputs(self):
        for t in (-1, 17, np.nan):
            with self.assertRaises(ValueError):
                c2_periodic(t, 16, [0.5], [1])
        for energies, overlaps in (([], []), ([0], [1]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                c2_periodic(1, 16, energies, overlaps)
        for tau, T, nt in ((4, 3, 16), (1, 17, 16), (-1, 3, 16)):
            with self.assertRaises(ValueError):
                c3_forward(tau, T, nt, [1], [1], [1], [1], [[1]])
        with self.assertRaises(ValueError):
            c3_forward(1, 3, 16, [1], [1], [1], [1], [1])


class KinematicsTests(unittest.TestCase):
    def test_momentum_units_and_shape(self):
        assert_allclose(lattice_momentum([[1, -2, 0]], [16, 32, 64], unit="lattice"), [[np.pi/8, -np.pi/8, 0]])
        assert_allclose(lattice_momentum([1, 0, 0], 32, unit="GeV", spacing=0.5), [np.pi/8, 0, 0])
        for kwargs in ({"unit":"fm"}, {"unit":"GeV"}, {"unit":"lattice", "spacing":1}):
            with self.assertRaises(ValueError):
                lattice_momentum([1, 0, 0], 32, **kwargs)
        with self.assertRaises(ValueError):
            lattice_momentum([0.5, 0, 0], 32, unit="lattice")

    def test_q2_metric(self):
        self.assertAlmostEqual(q_squared(2, [0,0,0], 1, [0.5,0,0], unit="GeV"), 0.75)
        self.assertAlmostEqual(q_squared(1, [1,0,0], 1, [-1,0,0], unit="lattice"), -4)

    @staticmethod
    def sample(pi=(0,0,0), pf=(0.3,0.2,0), mi=2., mf=0.5, fp=0.7, fm=-0.2):
        ei, ef = np.sqrt(mi**2+np.dot(pi,pi)), np.sqrt(mf**2+np.dot(pf,pf))
        p_i, p_f = np.r_[ei,pi], np.r_[ef,pf]
        current = (p_i+p_f)*fp+(p_i-p_f)*fm
        return mi,mf,ei,pi,ef,pf,current

    def test_pp_round_trip_moving_initial_and_final(self):
        result = pp_form_factors(*self.sample(pi=(0.2,-0.1,0.4)), unit="GeV")
        self.assertAlmostEqual(result.fplus, 0.7)
        self.assertAlmostEqual(result.fminus, -0.2)
        self.assertAlmostEqual(result.fzero, 0.7-result.q2*0.2/3.75)

    def test_pp_q2_zero_constraint_without_division_by_q2(self):
        # Exact binary numbers: Ei=2, Ef=1.25, |pf|=.75, Mf=1.
        args = self.sample(pf=(0.75,0,0), mf=1.)
        result = pp_form_factors(*args, unit="GeV")
        self.assertEqual(result.q2, 0)
        self.assertEqual(result.fzero, result.fplus)

    def test_pp_unit_scaling_invariance(self):
        args = self.sample()
        scale = 0.4
        scaled = [np.asarray(v)*scale for v in args]
        gev = pp_form_factors(*args, unit="GeV")
        lat = pp_form_factors(*scaled, unit="lattice")
        assert_allclose([gev.fplus, gev.fminus, gev.fzero], [lat.fplus, lat.fminus, lat.fzero])
        self.assertAlmostEqual(lat.q2, gev.q2*scale**2)

    def test_pp_rejects_degenerate_and_inconsistent_inputs(self):
        for args in (self.sample(pf=(0,0,0)), self.sample(mi=1,mf=1)):
            with self.assertRaises(ValueError):
                pp_form_factors(*args, unit="GeV")
        args = list(self.sample())
        args[2] += 0.1
        with self.assertRaises(ValueError):
            pp_form_factors(*args, unit="GeV")
        args = list(self.sample())
        args[-1][3] = 0.3
        with self.assertRaises(ValueError):
            pp_form_factors(*args, unit="GeV")
        args[-1] = args[-1].astype(complex)
        with self.assertRaises(ValueError):
            pp_form_factors(*args, unit="GeV")


class ZTests(unittest.TestCase):
    def test_z_reference_and_inverse(self):
        q2 = np.array([-1,0,1,3])
        z = z_map(q2, tplus=9, t0=1)
        assert_allclose(9-8*((1+z)/(1-z))**2, q2, atol=1e-14)
        self.assertEqual(z_map(1, tplus=9, t0=1), 0)

    def test_pole_and_explicit_no_pole(self):
        q2 = np.array([0., 1.])
        z = z_map(q2, tplus=9, t0=0)
        assert_allclose(pole_z_series(q2,[1,2,3],tplus=9,t0=0,pole_mass=2), (1+2*z+3*z*z)/(1-q2/4))
        assert_allclose(pole_z_series(q2,[1,2],tplus=9,t0=0,pole_mass=None), 1+2*z)

    def test_z_rejects_cut_and_pole(self):
        for q, tp, t0 in ((9,9,0),(10,9,0),(1,9,9),(0,-1,-2)):
            with self.assertRaises(ValueError):
                z_map(q,tplus=tp,t0=t0)
        with self.assertRaises(ValueError):
            pole_z_series(4,[1],tplus=9,t0=0,pole_mass=2)


class LegacyEvidenceReconstructionTests(unittest.TestCase):
    def test_explicit_euclidean_gamma_algebra(self):
        # Manually transcribed literals from trace_ST lines 25-29, no import.
        gamma = [
            1j*np.array([[0,0,0,-1],[0,0,-1,0],[0,1,0,0],[1,0,0,0]]),
            np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]),
            1j*np.array([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]]),
            np.diag([1,1,-1,-1]),
        ]
        g5 = -np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
        for a, ga in enumerate(gamma):
            assert_allclose(ga.conj().T, ga)
            assert_allclose(ga@g5+g5@ga, np.zeros((4,4)))
            for b, gb in enumerate(gamma):
                assert_allclose(ga@gb+gb@ga, 2*np.eye(4)*(a==b))
        assert_allclose(gamma[0]@gamma[1]@gamma[2]@gamma[3], g5)

    def test_time_orders_differ(self):
        nt, x0, T = 8, 6, 3
        start = (x0+T)%nt
        reverse = list(range(nt))[start::-1]+list(range(nt))[:start:-1]
        forward = list(range(nt))[start:]+list(range(start))
        self.assertEqual(reverse, [(x0+T-j)%nt for j in range(nt)])
        self.assertEqual(forward, [(x0+T+j)%nt for j in range(nt)])
        self.assertNotEqual(reverse, forward)

    def test_legacy_second_exponent_is_not_generic_thermal_model(self):
        # Independent evaluation of the displayed formula, not old code.
        ei, ef, T, tau = 0.3, 0.7, 8, 3
        second = lambda nt: np.exp(-ef*(T-nt+tau)-ei*(nt-tau))
        self.assertAlmostEqual(second(64)/second(32), np.exp((ef-ei)*32))


if __name__ == "__main__":
    unittest.main()
