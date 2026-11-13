from copy import deepcopy
import hashlib
import pickle
import unittest
import numpy as np
from numpy.testing import assert_allclose
from fixture_store import new_artifact_dir, create_store
from lqcd_analysis.joint_analysis import fit_joint_c2_c3, joint_spectral_model, joint_spectral_jacobian
from lqcd_analysis.physics import c2_periodic, c3_forward
from lqcd_analysis.fitting import FitError, correlated_fit


def joint_fixture():
    """Persistent, sealed SQLite with correlated synthetic source measurements."""
    nt,T=12,6
    x=np.arange(nt)
    ei,zi,ef,zf,m=.4,1.1,.25,.8,.7
    c3=np.zeros(nt)
    c3[:T+1]=c3_forward(np.arange(T+1),T,nt,[ei],[ef],[zi],[zf],[[m]]).real
    truths=np.stack([c2_periodic(x,nt,[ei],[zi]),c2_periodic(x,nt,[ef],[zf]),c3])
    rng=np.random.default_rng(91)
    half=rng.normal(0,2e-4,size=(20,3,nt))
    noise=np.concatenate([half,-half])
    files=[]; rows={name:[] for name in ("initial","final","three_point")}
    for cfg in range(40):
        config=str(1000+cfg)
        # Unequal source counts: first 20 configs have two sources, next 20 one.
        # Within each config, the extra source offsets cancel exactly.
        sources=(0,1) if cfg<20 else (0,)
        for t0 in sources:
            shift=(1 if t0==0 else -1)*1e-4 if cfg<20 else 0
            for i,name in enumerate(rows):
                y=truths[i]+noise[cfg,i]+shift
                if name=="initial":
                    path=f"initial/2pt_{config}_t{t0}_m0.100000_m0.400000_mom000.bin"; key="g5_g5"
                    arr=np.stack([y,np.zeros(nt)])
                elif name=="final":
                    path=f"final/2pt_{config}_t{t0}_m0.100000_m0.200000_mom000.bin"; key="g5_g5"
                    arr=np.stack([y,np.zeros(nt)])
                else:
                    path=f"3pt_{config}_t{t0}_T6_m0.400000_m0.200000_g5_m0.100000_g5_mom000_000.bin";key="g4"
                    # Known artificial phase i, undone explicitly by the fit.
                    arr=np.stack([np.zeros(nt),y])
                payload=pickle.dumps({key:arr},protocol=4)
                files.append(("synthetic",path,payload))
                rows[name].append(dict(root_id="synthetic",path=path,sha256=hashlib.sha256(payload).hexdigest(),
                                       config_id=config,source_id=f"t{t0}",source_time=t0))
    folder=new_artifact_dir("joint-sqlite")
    database,digest,_=create_store(folder,files)
    selections={}
    for name in rows:
        is3=name=="three_point"
        identity=dict(nt=nt,ensemble="synthetic",kind="3pt" if is3 else "2pt",physics_status="unconfirmed",
            ordered_masses=["0.400000","0.200000","0.100000"] if is3 else ["0.100000","0.400000" if name=="initial" else "0.200000"],
            source_operator="g5",sink_operator="g5",momentum_initial=[0,0,0],momentum_final=[0,0,0])
        if is3:identity.update(separation=T,current="g4")
        selections[name]=dict(version="lqcd.selection.v1",config_sha256=digest,stage="synthetic-validation",identity=identity,
            time_convention="sink-backward" if is3 else "source-forward",time_action="as-stored",
            convention_evidence="synthetic CPU-ST coordinate convention; artificial i current phase",
            component="complex" if is3 else "real",storage="split-ri",array_key="g4" if is3 else "g5_g5",
            profile="savedata.pickle.numeric.v1",allowed_momenta=[[0,0,0]],measurements=rows[name])
    sector=lambda e,z:dict(states=1,initial_energies=[e],initial_overlaps=[z],energy_gap_bounds=[[.01,1.]],overlap_bounds=[[.05,3.]])
    settings=dict(version="lqcd.joint-fit.v1",spectral_assumption="positive-diagonal-periodic",three_point_spectral_assumption="forward-only",
        model_initial=sector(.35,1.),model_final=sector(.3,.9),matrix_initial=[[.5]],matrix_bounds=[[[-5.,5.]]],
        windows=dict(initial=[1,2,3,4,5],final=[1,2,3,4,5],three_point=[1,2,3,4,5]),alignment="exact",
        configuration_order=[str(1000+i) for i in range(40)],bin_size=1,svd_rcond=1e-11,
        current_projection=dict(phase=[0.,-1.],scale=1.,evidence="Undo known synthetic i phase"),c3_time_mapping="cpu-st")
    return database,selections,settings,np.array([ei,zi,ef,zf,m])


class JointAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database,cls.selections,cls.settings,cls.truth=joint_fixture()

    def test_sqlite_joint_recovers_parameters_with_unequal_sources(self):
        result=fit_joint_c2_c3(self.database,self.selections,self.settings,verify_database_hash=True)
        assert_allclose(result["parameters"],self.truth,rtol=1e-7,atol=1e-8)
        self.assertEqual(result["configuration_count"],40)
        self.assertEqual(result["parameter_rank"],5)
        self.assertGreater(result["dof"],0)
        self.assertEqual(len(result["parameter_jackknife_samples"]),40)
        self.assertTrue(all(np.isfinite(result["standard_errors"])))

    def test_base_time_conversion_is_restored_once(self):
        converted=deepcopy(self.selections)
        converted["three_point"]["time_action"]="to-source-forward"
        result=fit_joint_c2_c3(self.database,converted,self.settings)
        assert_allclose(result["parameters"],self.truth,rtol=1e-7,atol=1e-8)
        self.assertIn("(T-tau)",result["time_transformations"][0])

    def test_ensemble_mass_identity_population_and_phase_mismatch_fail(self):
        for change in ("ensemble","mass","population","projection"):
            selections,settings=deepcopy(self.selections),deepcopy(self.settings)
            if change=="ensemble":selections["final"]["identity"]["ensemble"]="other"
            if change=="mass":selections["final"]["identity"]["ordered_masses"][1]="0.900000"
            if change=="population":selections["three_point"]["measurements"]=[r for r in selections["three_point"]["measurements"] if r["config_id"]!="1000"]
            if change=="projection":settings["current_projection"]["phase"]=[0.,2.]
            with self.subTest(change=change),self.assertRaises(ValueError):
                fit_joint_c2_c3(self.database,selections,settings)

    def test_contact_window_and_failed_fit_stop(self):
        settings=deepcopy(self.settings)
        settings["windows"]["three_point"]=[0,1,2,3]
        with self.assertRaises(ValueError):fit_joint_c2_c3(self.database,self.selections,settings)
        settings=deepcopy(self.settings);settings["max_nfev"]=1
        with self.assertRaises(FitError):fit_joint_c2_c3(self.database,self.selections,settings)

    def test_multistate_kernel_matrix_axis_order(self):
        ei,zi=np.array([.2,.5]),np.array([1.1,.7])
        ef,zf=np.array([.3,.8]),np.array([.8,.4])
        matrix=np.array([[.2,.3],[.7,.9]])
        p=np.r_[np.log([.2,.3]),np.log(zi),np.log([.3,.5]),np.log(zf),matrix.ravel()]
        actual=joint_spectral_model([[0,2],[1,3],[2,1],[2,4]],p,nt=12,separation=6,n_initial=2,n_final=2)
        expected=[c2_periodic(2,12,ei,zi),c2_periodic(3,12,ef,zf)]
        for t in (1,4):
            expected.append(sum(zf[j]*zi[i]*matrix[j,i]/(4*ef[j]*ei[i])*np.exp(-ef[j]*(6-t)-ei[i]*t) for j in range(2) for i in range(2)))
        assert_allclose(actual,expected)

    def test_analytic_jacobian_matches_central_difference(self):
        p=np.r_[np.log([.2,.3]),np.log([1.1,.7]),np.log([.3,.4]),np.log([.8,.4]),[.2,.3,.7,.9]]
        x=np.array([[0,2],[1,3],[2,1],[2,4]])
        settings=dict(nt=12,separation=6,n_initial=2,n_final=2)
        actual=joint_spectral_jacobian(x,p,**settings)
        expected=np.zeros_like(actual)
        for i in range(len(p)):
            step=np.zeros_like(p);step[i]=1e-5
            expected[:,i]=(joint_spectral_model(x,p+step,**settings)-joint_spectral_model(x,p-step,**settings))/(2e-5)
        assert_allclose(actual,expected,rtol=2e-8,atol=2e-10)

    def test_coincident_matrix_exponents_are_not_falsely_identifiable(self):
        # Ei1-Ei0 = Ef1-Ef0, so M00 and M11 have the same tau dependence.
        p=np.r_[np.log([.3,.4]),np.log([1.,.5]),np.log([.2,.4]),np.log([.8,.4]),[.2,.3,.7,.9]]
        x=np.array([[k,t] for k in range(3) for t in range(1,11)])
        settings=dict(nt=24,separation=12,n_initial=2,n_final=2)
        model=lambda x,p:joint_spectral_model(x,p,**settings)
        jac=lambda x,p:joint_spectral_jacobian(x,p,**settings)
        y=model(x,p)
        fit=correlated_fit(x,y,np.eye(len(y)),model,p,bounds=(p-1,p+1),model_jac=jac,svd_rcond=1e-11)
        self.assertLess(fit.parameter_rank,len(p))
        self.assertIsNone(fit.pvalue)
        self.assertIsNone(fit.parameter_covariance)


if __name__=="__main__":unittest.main()
