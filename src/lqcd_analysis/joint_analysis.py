"""SQLite-backed correlated C2(initial)+C2(final)+C3 spectral fits.

All inputs are the same ensemble and configuration population. Source averages
precede equal-size configuration bins. The current projection and time mapping
are explicit; no Euclidean-to-Minkowski or P->V/V->P extraction is inferred.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import numpy as np
from .database import CorrelatorStore
from .selection import load_channel, load_selection
from .statistics import ConfigSamples, average_sources, align_channels, bin_configurations, jackknife, jackknife_covariance
from .fitting import fit_jackknife, FitError
from .physics import c2_periodic, c3_forward


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _sector(spec):
    n=spec.get("states")
    _need(isinstance(n,int) and not isinstance(n,bool) and 1<=n<=4,"Choose 1..4 states per sector")
    e=np.asarray(spec.get("initial_energies"),dtype=float)
    z=np.asarray(spec.get("initial_overlaps"),dtype=float)
    _need(e.shape==z.shape==(n,) and np.all(np.isfinite(e)) and np.all(np.isfinite(z)),"State parameter shape/finiteness mismatch")
    gaps=np.diff(np.r_[0.,e])
    _need(np.all(gaps>0) and np.all(z>0),"Initial energies must increase and overlaps must be positive")
    bounds=np.asarray(spec.get("energy_gap_bounds",[])+spec.get("overlap_bounds",[]),dtype=float)
    _need(bounds.shape==(2*n,2) and np.all(np.isfinite(bounds)) and np.all(bounds[:,0]>0)
          and np.all(bounds[:,1]>bounds[:,0]),"Provide positive finite gap and overlap bounds")
    p=np.log(np.r_[gaps,z]); lo,hi=np.log(bounds[:,0]),np.log(bounds[:,1])
    _need(np.all(p>lo) and np.all(p<hi),"Initial sector parameters must be strictly inside bounds")
    return n,p,lo,hi


def unpack_joint_parameters(parameters, n_initial, n_final):
    """Ordered energies, positive overlaps, real M[final,initial]."""
    p=np.asarray(parameters)
    ni,nf=n_initial,n_final
    _need(p.shape==(2*ni+2*nf+ni*nf,),"Joint parameter count mismatch")
    ei=np.cumsum(np.exp(p[:ni])); zi=np.exp(p[ni:2*ni])
    o=2*ni
    ef=np.cumsum(np.exp(p[o:o+nf])); zf=np.exp(p[o+nf:o+2*nf])
    matrix=p[o+2*nf:].reshape(nf,ni)
    return ei,zi,ef,zf,matrix


def joint_spectral_model(x, parameters, *, nt, separation, n_initial, n_final):
    """Rows of x are [channel_number(0,1,2), physical_time]."""
    x=np.asarray(x)
    _need(x.ndim==2 and x.shape[1]==2 and np.all(np.isfinite(x)),"Joint x must have channel/time columns")
    _need(np.all(np.isin(x[:,0],[0,1,2])),"Unknown joint channel number")
    ei,zi,ef,zf,matrix=unpack_joint_parameters(parameters,n_initial,n_final)
    result=np.empty(len(x))
    for number in range(3):
        mask=x[:,0]==number
        t=x[mask,1]
        if number==0:
            result[mask]=c2_periodic(t,nt,ei,zi)
        elif number==1:
            result[mask]=c2_periodic(t,nt,ef,zf)
        else:
            result[mask]=c3_forward(t,separation,nt,ei,ef,zi,zf,matrix).real
    return result


def joint_spectral_jacobian(x, parameters, *, nt, separation, n_initial, n_final):
    """Analytic derivative in log-gap/log-overlap/M coordinates.

    In particular the M block is exact. Finite-difference noise must not turn
    coincident spectral exponents into artificial identifiable matrix elements.
    """
    x=np.asarray(x,dtype=float);p=np.asarray(parameters,dtype=float)
    ei,zi,ef,zf,matrix=unpack_joint_parameters(p,n_initial,n_final)
    ni,nf=n_initial,n_final;offset=2*ni
    chain_i=np.tril(np.ones((ni,ni)))*np.exp(p[:ni])[None,:]
    chain_f=np.tril(np.ones((nf,nf)))*np.exp(p[offset:offset+nf])[None,:]
    jac=np.zeros((len(x),len(p)))
    for row,(kind,t) in enumerate(x):
        if kind in (0,1):
            e,z,chain,start=(ei,zi,chain_i,0) if kind==0 else (ef,zf,chain_f,offset)
            n=len(e)
            forward=np.exp(-e*t);backward=np.exp(-e*(nt-t))
            terms=z*z/(2*e)*(forward+backward)
            de=z*z/2*((-t*forward-(nt-t)*backward)/e-(forward+backward)/(e*e))
            jac[row,start:start+n]=de@chain
            jac[row,start+n:start+2*n]=2*terms
        elif kind==2:
            kernel=zf[:,None]*zi[None,:]/(4*ef[:,None]*ei[None,:])*np.exp(-ef[:,None]*(separation-t)-ei[None,:]*t)
            terms=kernel*matrix
            jac[row,:ni]=(terms.sum(axis=0)*(-t-1/ei))@chain_i
            jac[row,ni:2*ni]=terms.sum(axis=0)
            jac[row,offset:offset+nf]=(terms.sum(axis=1)*(-(separation-t)-1/ef))@chain_f
            jac[row,offset+nf:offset+2*nf]=terms.sum(axis=1)
            jac[row,offset+2*nf:]=kernel.ravel()
        else:
            raise ValueError("Unknown joint channel number")
    return jac


def _physical(p,ni,nf):
    ei,zi,ef,zf,m=unpack_joint_parameters(p,ni,nf)
    return np.r_[ei,zi,ef,zf,m.ravel()]


def fit_joint_c2_c3(database, selections, settings, *, verify_database_hash=False):
    """Fit explicit manifests {initial,final,three_point} from a sealed SQLite.

    settings requires model_initial/model_final, matrix_initial/matrix_bounds,
    windows, alignment, configuration_order, bin_size, svd_rcond,
    current_projection={phase:[real,imag],scale,evidence}, c3_time_mapping,
    and explicit spectral_assumption/three_point_spectral_assumption.
    """
    names=("initial","final","three_point")
    _need(settings.get("version")=="lqcd.joint-fit.v1","Unsupported joint fit settings")
    _need(set(selections)==set(names),"Supply initial/final/three_point selections")
    _need(settings.get("spectral_assumption")=="positive-diagonal-periodic"
          and settings.get("three_point_spectral_assumption")=="forward-only",
          "Explicit C2 positive-diagonal-periodic and C3 forward-only assumptions required")
    digests={s.get("config_sha256") for s in selections.values()}
    _need(len(digests)==1,"Joint selections must refer to one migration configuration")
    with CorrelatorStore(database,verify_database_hash=verify_database_hash,
                         expected_config_sha256=next(iter(digests))) as store:
        channels={name:load_channel(store,selections[name]) for name in names}
    ids={name:c.identity for name,c in channels.items()}
    _need(len({i["ensemble"] for i in ids.values()})==1 and len({i["nt"] for i in ids.values()})==1,
          "Joint channels must have the same ensemble and temporal extent")
    _need(ids["initial"]["kind"]==ids["final"]["kind"]=="2pt" and ids["three_point"]["kind"]=="3pt","Wrong joint channel kinds")
    for role in ("initial","final"):
        _need(ids[role]["source_operator"]==ids[role]["sink_operator"],"C2 sectors require identical source/sink operators")
        _need(selections[role]["component"]=="real" and channels[role].provenance["output_time_convention"]=="source-forward",
              "C2 sectors need explicitly projected real source-forward data")
    m1,m2,m3=map(Decimal,ids["three_point"]["ordered_masses"])
    _need(list(map(Decimal,ids["initial"]["ordered_masses"]))==[m3,m1]
          and list(map(Decimal,ids["final"]["ordered_masses"]))==[m3,m2],
          "Mass routing must be C3=(m_initial,m_final,m_spectator), C2=(m_spectator,m_active)")
    for role,opkey in (("initial","source_operator"),("final","sink_operator")):
        _need(ids[role]["source_operator"]==ids["three_point"][opkey],"C2 and C3 interpolating operators disagree")
        _need(ids[role]["momentum_initial"]==ids[role]["momentum_final"]==ids["three_point"]["momentum_"+role],"C2 and C3 momenta disagree")
    _need((m1,ids["initial"]["source_operator"],tuple(ids["initial"]["momentum_initial"])) !=
          (m2,ids["final"]["source_operator"],tuple(ids["final"]["momentum_initial"])),
          "Identical initial/final C2 requires a shared-parameter model, not duplicate independent sectors")
    nt=ids["initial"]["nt"]; T=ids["three_point"]["separation"]
    _need(0<T<nt,"Forward C3 fit requires 0<T<NT")
    c3=channels["three_point"]
    mapping=settings.get("c3_time_mapping")
    output_time=c3.provenance["output_time_convention"]
    values={k:np.asarray(c.values) for k,c in channels.items()}
    transformations=[]
    if mapping=="initial-forward":
        _need(output_time=="initial-forward","Declared initial-forward C3 data required")
    elif mapping=="cpu-st":
        _need(selections["three_point"]["time_convention"]=="sink-backward","CPU ST mapping requires known sink-backward writer convention")
        if output_time=="source-forward":
            values["three_point"]=values["three_point"][:,(T-np.arange(nt))%nt]
            transformations.append("base-source-forward to physical initial tau: index=(T-tau)%NT")
        else:
            _need(output_time=="sink-backward","Unsupported CPU ST output time convention")
            transformations.append("CPU ST stored j is physical initial-to-current tau; no reorder")
    else:
        raise ValueError("Choose c3_time_mapping initial-forward or cpu-st explicitly")
    _need(selections["three_point"]["component"]=="complex","Keep complex C3 until its explicit current projection")
    projection=settings.get("current_projection",{})
    phase=np.asarray(projection.get("phase"),dtype=float)
    scale=projection.get("scale")
    _need(phase.shape==(2,) and np.all(np.isfinite(phase)) and np.isclose(np.dot(phase,phase),1,rtol=1e-10,atol=1e-12),"Current phase must be an explicit unit complex number [real,imag]")
    _need(isinstance(scale,(int,float)) and np.isfinite(scale) and scale!=0 and isinstance(projection.get("evidence"),str)
          and projection["evidence"].strip(),"Current normalization scale and convention evidence required")
    values["three_point"]=np.real(complex(*phase)*values["three_point"])*scale
    # Membership is intersected/validated BEFORE averaging any sources.
    alignment=settings.get("alignment")
    _need(alignment in {"exact","intersection"},"Choose exact/intersection configuration alignment explicitly")
    populations={k:set(c.config_ids) for k,c in channels.items()}
    if alignment=="exact":
        _need(all(s==populations["initial"] for s in populations.values()),"Joint configuration populations differ")
    common=set.intersection(*populations.values())
    _need(common,"No shared configurations")
    dropped={k:sorted(v-common) for k,v in populations.items()}
    averages={}
    for name,c in channels.items():
        chosen=[i for i,cfg in enumerate(c.config_ids) if cfg in common]
        averages[name]=average_sources([c.config_ids[i] for i in chosen],values[name][chosen],[c.source_ids[i] for i in chosen])
    averages=align_channels(averages,"exact").channels
    order=settings.get("configuration_order")
    _need(isinstance(order,list) and all(isinstance(v,str) for v in order) and len(order)==len(set(order)) and set(order)==common,
          "Provide the exact retained configuration order for binning")
    binned={}
    for name,c in averages.items():
        pos={v:i for i,v in enumerate(c.config_ids)}
        binned[name]=bin_configurations(ConfigSamples(tuple(order),c.values[[pos[v] for v in order]]),settings.get("bin_size"))
    windows=settings.get("windows",{})
    _need(set(windows)==set(names),"Choose a fixed window for every channel")
    pieces=[]; coordinates=[]
    for number,name in enumerate(names):
        w=np.asarray(windows[name])
        _need(w.ndim==1 and w.dtype.kind in "iu" and len(w)>0 and np.all(np.diff(w)>0),"Windows must be nonempty strictly increasing integer lists")
        upper=T if name=="three_point" else nt
        lower=1 if name=="three_point" else 0
        _need(np.all((w>=lower)&(w<upper)),"Window outside domain; exclude C3 contact endpoints 0 and T")
        pieces.append(binned[name].values[:,w])
        coordinates.extend([[number,int(t)] for t in w])
    joined=ConfigSamples(binned["initial"].config_ids,np.concatenate(pieces,axis=1))
    jk=jackknife(joined)
    ni,pi,li,ui=_sector(settings["model_initial"])
    nf,pf,lf,uf=_sector(settings["model_final"])
    m=np.asarray(settings["matrix_initial"],dtype=float)
    mb=np.asarray(settings["matrix_bounds"],dtype=float)
    _need(m.shape==(nf,ni) and mb.shape==(nf,ni,2) and np.all(np.isfinite(m)) and np.all(np.isfinite(mb))
          and np.all(mb[:,:,0]<m) and np.all(m<mb[:,:,1]),"Real M[final,initial] initial values/bounds mismatch")
    p=np.r_[pi,pf,m.ravel()]; lo=np.r_[li,lf,mb[:,:,0].ravel()]; hi=np.r_[ui,uf,mb[:,:,1].ravel()]
    _need("svd_rcond" in settings,"Explicit covariance SVD cutoff required")
    def model(x,p):
        return joint_spectral_model(x,p,nt=nt,separation=T,n_initial=ni,n_final=nf)
    def model_jac(x,p):
        return joint_spectral_jacobian(x,p,nt=nt,separation=T,n_initial=ni,n_final=nf)
    fit=fit_jackknife(np.asarray(coordinates),jk,model,p,window=np.arange(len(coordinates)),bounds=(lo,hi),
                      svd_rcond=settings["svd_rcond"],max_nfev=settings.get("max_nfev"),model_jac=model_jac)
    _need(fit.full_fit.parameter_rank==len(p) and fit.full_fit.dof>0,"Insufficient identifiable joint information")
    if any(f.parameter_rank!=len(p) for f in fit.sample_fits):
        raise FitError("A joint jackknife sample has unidentifiable parameters")
    central=_physical(fit.full_fit.parameters,ni,nf)
    replicates=np.stack([_physical(v,ni,nf) for v in fit.parameter_samples])
    covariance=jackknife_covariance(replicates)
    active=bool(np.any(fit.full_fit.active_bounds) or any(np.any(f.active_bounds) for f in fit.sample_fits))
    labels=([f"energy_initial_{i}" for i in range(ni)]+[f"overlap_initial_{i}" for i in range(ni)]
            +[f"energy_final_{i}" for i in range(nf)]+[f"overlap_final_{i}" for i in range(nf)]
            +[f"matrix_final_{j}_initial_{i}" for j in range(nf) for i in range(ni)])
    return {"version":"lqcd.joint-fit-result.v1","status":"requires-bound-review" if active else "validated-numerics",
        "scientific_status":"bare projected current spectral fit; no automatic Minkowski conversion or spinful form factors",
        "identities":ids,"fit_settings":settings,"provenance":{k:c.provenance for k,c in channels.items()},
        "time_transformations":transformations,"parameter_order":labels,"parameters":central.tolist(),
        "standard_errors":np.sqrt(np.diag(covariance)).tolist(),"parameter_covariance":covariance.tolist(),
        "parameter_jackknife_samples":replicates.tolist(),"matrix_elements":central[-ni*nf:].reshape(nf,ni).tolist(),
        "configuration_count":len(order),"bin_memberships":[list(x) for x in joined.config_ids],
        "dropped_configuration_ids":dropped,"chi2":fit.full_fit.chi2,"dof":fit.full_fit.dof,
        "covariance_rank":fit.full_fit.rank,"parameter_rank":fit.full_fit.parameter_rank,"active_bounds":active,
        "pvalue":fit.full_fit.pvalue,"pvalue_note":fit.full_fit.pvalue_note,
        "covariance_eigenvalues":fit.full_fit.covariance_eigenvalues.tolist(),"covariance_cutoff":fit.full_fit.svd_cutoff,
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run_joint_file(path, *, verify_database_hash=False):
    path=Path(path).resolve()
    settings=json.loads(path.read_text(encoding="utf-8"))
    def local(value):
        value=Path(value)
        return value if value.is_absolute() else path.parent/value
    selections={k:load_selection(local(v)) for k,v in settings["selections"].items()}
    return fit_joint_c2_c3(local(settings["database"]),selections,settings,verify_database_hash=verify_database_hash)
