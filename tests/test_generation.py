import json
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from fixture_store import new_artifact_dir
from lqcd_analysis.generation import (
    Geometry, source_field, gamma, sequential_source, fourier_phase, contract_two_point,
    contract_three_point, solve_reference, write_xqcd, read_xqcd, run_generation,
    load_product, correlator_records, solve_xqcd,
)


class GenerationTests(unittest.TestCase):
    def test_gamma_clifford_and_hermiticity(self):
        for i in range(1,5):
            a=gamma(f"g{i}")
            assert_allclose(a.conj().T,a)
            for j in range(1,5):
                b=gamma(f"g{j}")
                assert_allclose(a@b+b@a,2*np.eye(4) if i==j else np.zeros((4,4)))
            assert_allclose(a@gamma("g5")+gamma("g5")@a,0)
        assert_allclose(gamma("g5")@gamma("g5"),np.eye(4))
        # CPU ST main constructs sigmaij with gatoGa('gi.gj'), without i.
        assert_allclose(gamma("sigma12"),gamma("g1")@gamma("g2"))

    def test_point_and_reproducible_z3_spin_color_dilution(self):
        g=Geometry(4,2,2,2)
        p=source_field(g,kind="point",position=[1,0,1,2])
        self.assertEqual(np.count_nonzero(p.arrays["source"]),12)
        z=source_field(g,kind="z3-grid",position=[0,0,0,1],seed=11,grid_spacing=[1,1,1])
        repeat=source_field(g,kind="z3-grid",position=[0,0,0,1],seed=11,grid_spacing=[1,1,1])
        assert_array_equal(z.arrays["source"],repeat.arrays["source"])
        noise=z.arrays["source"][0,0,0,0,1]
        assert_allclose(noise**3,1,atol=2e-15)
        self.assertEqual(np.count_nonzero(z.arrays["source"]),12*8)
        with self.assertRaises(ValueError):
            source_field(g,kind="z3-grid",position=[0,0,0,1],seed=11,grid_spacing=[3,1,1])

    def test_raw_big_endian_axis_roundtrip_and_no_overwrite(self):
        g=Geometry(2,1,1,2)
        a=np.arange(np.prod(g.propagator_shape)).reshape(g.propagator_shape)*(1+2j)
        folder=new_artifact_dir("generation-raw")
        path=folder/"prop.raw"
        write_xqcd(path,a,g)
        assert_array_equal(read_xqcd(path,g),a)
        self.assertEqual(path.stat().st_size,a.size*16)
        with self.assertRaises(FileExistsError):
            write_xqcd(path,a,g)

    def test_dense_solve_true_residual_and_singular_failure(self):
        g=Geometry(2,1,1,1)
        src=source_field(g,kind="point",position=[0,0,0,0]).arrays["source"]
        d=2*np.eye(g.dimension)-.1*np.roll(np.eye(g.dimension),1,axis=1)
        result=solve_reference(d,src,g)
        assert_allclose(d@result.arrays["propagator"].reshape(12,-1).T,src.reshape(12,-1).T,atol=1e-14)
        self.assertLess(max(result.metadata["relative_residuals"]),1e-13)
        with self.assertRaises(Exception):
            solve_reference(np.zeros_like(d),src,g)

    def test_sequential_source_nonzero_momentum_and_time_wrap(self):
        g=Geometry(4,1,1,2)
        a=np.arange(np.prod(g.propagator_shape)).reshape(g.propagator_shape)+.2j
        product=sequential_source(a,g,source_position=[1,0,0,3],separation=2,
                                  initial_operator="g2",momentum_initial=[1,0,0])
        expected=np.zeros_like(a,dtype=complex)
        phase=fourier_phase(g,[1,0,0],[1,0,0,3])
        for ss in range(4):
            for sc in range(3):
                for ks in range(4):
                    for kc in range(3):
                        for x in range(2):
                            expected[ss,sc,ks,kc,1,0,0,x]=sum(gamma("g2")[ks,b]*a[ss,sc,b,kc,1,0,0,x] for b in range(4))*phase[0,0,x]
        assert_allclose(product.arrays["source"],expected)

    def test_zero_momentum_grid_estimator_has_source_average_normalization(self):
        g=Geometry(4,1,1,2)
        temporal=2.5*np.eye(4)-.2*np.roll(np.eye(4),1,axis=1)-.2*np.roll(np.eye(4),-1,axis=1)
        d=np.kron(np.eye(12),np.kron(temporal,np.eye(2)))
        estimates=[]
        for kind in ("point","z3-grid"):
            kwargs=dict(seed=5,grid_spacing=[1,1,1]) if kind=="z3-grid" else {}
            source=source_field(g,kind=kind,position=[0,0,0,0],**kwargs)
            prop=solve_reference(d,source.arrays["source"],g).arrays["propagator"]
            seq=sequential_source(prop,g,source_position=[0,0,0,0],separation=2,initial_operator="g5",momentum_initial=[0,0,0])
            seqprop=solve_reference(d,seq.arrays["source"],g).arrays["propagator"]
            c3=contract_three_point(prop,seqprop,g,source_position=[0,0,0,0],separation=2,current="g5",final_operator="g5",momentum_initial=[0,0,0],momentum_final=[0,0,0])
            estimates.append(c3/source.metadata["support_site_count"])
        # Spatially disconnected identical sites make noise cross terms zero.
        assert_allclose(estimates[0],estimates[1],atol=1e-14)

    def test_cpu_contractions_against_independent_explicit_spin_sum(self):
        g=Geometry(3,1,1,2)
        rng=np.random.default_rng(4)
        a=rng.normal(size=g.propagator_shape)+1j*rng.normal(size=g.propagator_shape)
        b=rng.normal(size=g.propagator_shape)+1j*rng.normal(size=g.propagator_shape)
        position=[1,0,0,2]
        phase=fourier_phase(g,[1,0,0],position)
        left,right=gamma("g5")@gamma("g1"),gamma("g2")@gamma("g5")
        raw=np.zeros(3,dtype=complex)
        for t in range(3):
            for x in range(2):
                for ss in range(4):
                    for us in range(4):
                        for ks in range(4):
                            for vs in range(4):
                                for sc in range(3):
                                    for kc in range(3):
                                        raw[t]+=a[ss,sc,ks,kc,t,0,0,x].conjugate()*b[us,sc,vs,kc,t,0,0,x]*left[ks,vs]*right[us,ss]*phase[0,0,x]
        c2=contract_two_point(a,b,g,source_position=position,source_operator="g2",sink_operator="g1",momentum=[1,0,0])
        c3=contract_three_point(a,b,g,source_position=position,separation=2,current="g1",final_operator="g2",
                                momentum_initial=[1,0,0],momentum_final=[0,0,0])
        assert_allclose(c2,raw[[2,0,1]],atol=1e-12)
        assert_allclose(c3,raw[[1,0,2]],atol=1e-12)

    def test_complete_reference_pipeline_and_product_integrity(self):
        folder=new_artifact_dir("generation-full")
        g=Geometry(4,1,1,1)
        operators={}
        for role,mass in (("initial",.4),("final",.2),("spectator",.1)):
            temporal=(2+mass)*np.eye(4)-.25*np.roll(np.eye(4),1,axis=1)-.25*np.roll(np.eye(4),-1,axis=1)
            path=folder/(role+".npy")
            with path.open("xb") as stream:np.save(stream,np.kron(np.eye(12),temporal))
            operators[role]=str(path)
        config=dict(version="lqcd.generation.v1",stage="full",writer_profile="cpu-sequential-reference.v1",
            scientific_mode="mathematical-validation",ensemble="synthetic",config_id="1",source_id="t0",geometry=dict(nt=4,nz=1,ny=1,nx=1),
            source=dict(kind="point",position=[0,0,0,0]),masses=dict(initial="0.400000",final="0.200000",spectator="0.100000"),
            channel=dict(initial_operator="g5",final_operator="g5",current="g4",momentum_initial=[0,0,0],momentum_final=[0,0,0],separation=2),
            backend=dict(profile="explicit-matrix-reference",operators=operators,residual_tolerance=1e-12))
        product=run_generation(config,folder/"output")
        loaded=load_product(folder/"output")
        assert_array_equal(loaded.arrays["c3"],product.arrays["c3"])
        records=correlator_records(product)
        self.assertEqual(len(records),3)
        self.assertEqual(records[2]["metadata"]["time_convention"],"initial-forward")
        self.assertEqual(records[2]["metadata"]["ordered_masses"],["0.400000","0.200000","0.100000"])
        self.assertEqual(product.metadata["backend_status"],"explicit-operator-reference")
        # Stage decomposition produces identical tensors without regenerating sources.
        source_config=dict(config,stage="source")
        run_generation(source_config,folder/"source-stage")
        solve_config=dict(config,stage="solve",input_product=str(folder/"source-stage"))
        run_generation(solve_config,folder/"solve-stage")
        contraction_config=dict(config,stage="contract",input_product=str(folder/"solve-stage"))
        staged=run_generation(contraction_config,folder/"contract-stage")
        assert_allclose(staged.arrays["c3"],product.arrays["c3"])
        with self.assertRaises(FileExistsError):run_generation(config,folder/"output")

    def test_xqcd_stdin_adapter_executes_test_double_and_checks_outputs(self):
        folder=new_artifact_dir("generation-backend-contract")
        # This is a process-contract test double, not a lattice action or solver.
        script=folder/"backend_double.py"
        script.write_text("import sys,pathlib\nf=dict(line.split(maxsplit=1) for line in sys.stdin.buffer.read().decode('utf-8').splitlines() if line.strip())\np=pathlib.Path(f['source_text00'])\nout=pathlib.Path(str(p)+'.read00.m'+f['masses:0'])\nwith out.open('xb') as s:s.write(p.read_bytes())\nfor i in range(12):print('TEST_RELRES 0.0')\n",encoding="utf-8")
        gauge=folder/"gauge.test-only"
        gauge.write_bytes(b"test-double-not-gauge")
        template=folder/"input.template"
        template.write_text("nx $nx\nny $ny\nnz $nz\nnt $nt\ncfg_start $config_id\ncfg_end $config_id\nnsource 1\nsource_type00 read_source\nsource_text00 $source\nmasses:length 1\nmasses:0 $mass\nreload_kentucky_be $gauge\nsave_propagator 1\nrotate_zt 0\nkappa 0.2\nboundary_conditions 1 1 1 -1\ncg_error 1e-12\nmaxiter 20\npoly_error 1e-12\none_minus_halfD 1\nHwnoeigen 0\nov_noeigen 0\nput_bak_low 0\nprop_name $workdir/prop\nHwilson_name $workdir/hw\nov_evecname $workdir/ov\n",encoding="utf-8")
        backend=dict(profile="xqcd.overlap-inverter-stdin.v1",argv=[sys.executable,str(script)],input_template=str(template),
                     action_evidence="test double, no production action",residual_pattern=r"TEST_RELRES ([0-9.e+-]+)",
                     residual_definition="test contract value only",residual_evidence="test-double script",
                     residual_tolerance=1e-10,expected_residual_count=12,timeout_seconds=20)
        g=Geometry(2,1,1,1)
        source=source_field(g,kind="point",position=[0,0,0,0]).arrays["source"]
        result=solve_xqcd(source,g,mass=.2,config_id=1,gauge=gauge,backend=backend,workdir=folder/"run")
        assert_array_equal(result.arrays["propagator"],source)
        self.assertEqual(len(result.metadata["residuals"]),12)
        broken=dict(backend,residual_pattern=r"ABSENT ([0-9.e+-]+)")
        with self.assertRaisesRegex(ValueError,"residual evidence"):
            solve_xqcd(source,g,mass=.2,config_id=1,gauge=gauge,backend=broken,workdir=folder/"missing-residual")


if __name__=="__main__":unittest.main()
