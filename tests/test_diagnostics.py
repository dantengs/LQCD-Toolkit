import unittest
import numpy as np
from numpy.testing import assert_allclose
from lqcd_analysis.diagnostics import effective_energy, effective_energy_jackknife, EffectiveEnergySampleError
from lqcd_analysis.statistics import ConfigSamples, jackknife


class DiagnosticsTests(unittest.TestCase):
    def test_forward_exponential_has_positive_log_energy_and_correct_times(self):
        dt=.5; origin=3.
        data=2*np.exp(-.7*(origin+np.arange(6)*dt))
        result=effective_energy(data,method="log",spacing=dt,time_origin=origin)
        assert_allclose(result.values,.7)
        assert_allclose(result.times,[3.25,3.75,4.25,4.75,5.25])
        # The original inverted ratio has the opposite sign.
        assert_allclose(np.log(data[1:]/data[:-1])/dt,-.7)

    def test_periodic_single_state_cosh_energy(self):
        t=np.arange(12);e=.35
        c=1.7*(np.exp(-e*t)+np.exp(-e*(12-t)))
        result=effective_energy(c,method="cosh")
        assert_allclose(result.values,e,atol=1e-14)
        assert_allclose(result.times,np.arange(1,11))
        # Log effective energy has thermal/backward contamination near midpoint.
        self.assertGreater(np.max(np.abs(effective_energy(c,method="log").values-e)),.1)

    def test_no_sign_projection_or_cosh_domain_clipping(self):
        for data in ([1.,0.],[1.,-2.],[1.,np.nan],[1.,np.inf],[1.+0j,2.+0j]):
            with self.subTest(data=data),self.assertRaises(ValueError):
                effective_energy(data,method="log")
        with self.assertRaisesRegex(ValueError,"no clipping"):
            effective_energy([1.,2.,1.],method="cosh")
        with self.assertRaises(ValueError):effective_energy([2.,1.],method="cosh")
        with self.assertRaises(ValueError):effective_energy([2.,1.],method="log",spacing=0.)

    def test_stable_log_difference_without_ratio_overflow(self):
        result=effective_energy([1e300,1e-300],method="log")
        assert_allclose(result.values,[600*np.log(10)])

    def test_full_nonlinear_jackknife_and_cross_time_covariance(self):
        raw=np.array([[10.,5.,2.],[8.,5.,3.],[12.,7.,4.],[9.,4.,3.]])
        data=jackknife(ConfigSamples(tuple("abcd"),raw))
        result=effective_energy_jackknife(data,method="log")
        # Independent ratio calculation from explicit deleted-configuration sums.
        replicas=[]
        for i in range(4):
            c=np.mean(raw[[j for j in range(4) if j!=i]],axis=0)
            replicas.append(np.log(c[:-1]/c[1:]))
        replicas=np.array(replicas)
        centered=replicas-replicas.mean(axis=0)
        expected_cov=.75*sum(np.outer(v,v) for v in centered)
        assert_allclose(result.samples,replicas)
        assert_allclose(result.covariance,expected_cov)
        assert_allclose(result.mean,np.log(raw.mean(axis=0)[:-1]/raw.mean(axis=0)[1:]))
        assert_allclose(result.bias_corrected_mean,4*result.mean-3*replicas.mean(axis=0))
        self.assertEqual(result.config_ids,tuple("abcd"))
        self.assertNotEqual(float(result.covariance[0,1]),0.)

    def test_one_invalid_jackknife_sample_reports_deleted_identity(self):
        data=jackknife(ConfigSamples(("large","small1","small2"),[[10.,10.],[-2.,1.],[-2.,1.]]))
        self.assertTrue(np.all(data.mean>0))
        with self.assertRaises(EffectiveEnergySampleError) as caught:
            effective_energy_jackknife(data,method="log")
        self.assertEqual(caught.exception.sample_index,0)
        self.assertEqual(caught.exception.config_id,"large")


if __name__=="__main__":unittest.main()
