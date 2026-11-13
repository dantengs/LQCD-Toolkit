"""Explicit CPU-reference source -> solve -> sequential-source -> contraction.

This is not an implementation of the production overlap action. The production
adapter runs an explicitly supplied XQCD executable with its documented stdin
contract; the small dense-operator backend is a mathematical reference only.
Canonical tensor axes are hidden behind the xqcd-be-f64.v1 I/O profile.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from string import Template
import hashlib
import json
import re
import shutil
import subprocess
import numpy as np
from scipy.linalg import solve


PROFILE = "cpu-sequential-reference.v1"
PROP_AXES = ["source_spin", "source_color", "sink_spin", "sink_color", "t", "z", "y", "x"]


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _finite(value, name):
    a = np.asarray(value)
    _need(a.dtype.kind in "iufc" and np.all(np.isfinite(a)), f"{name} must be finite numeric data")
    return a


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps([a.dtype.str, list(a.shape)]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Geometry:
    nt: int
    nz: int
    ny: int
    nx: int

    def __post_init__(self):
        _need(all(_integer(n) and n > 0 for n in (self.nt, self.nz, self.ny, self.nx)), "Positive integer lattice extents required")

    @property
    def shape(self):
        return self.nt, self.nz, self.ny, self.nx

    @property
    def propagator_shape(self):
        return (4, 3, 4, 3) + self.shape

    @property
    def dimension(self):
        return 12 * int(np.prod(self.shape))


@dataclass(frozen=True)
class NumericProduct:
    kind: str
    arrays: dict[str, np.ndarray]
    metadata: dict


def _prop(value, geometry):
    value = _finite(value, "propagator/source")
    _need(value.shape == geometry.propagator_shape, "Tensor does not match the declared XQCD geometry")
    return np.asarray(value, dtype=np.complex128)


def gamma(label):
    """CPU writer's basis; legacy sigma_ij = gi*gj = [gi,gj]/2 (NO factor i)."""
    g = {
        "Id": np.eye(4, dtype=complex),
        "g1": 1j*np.array([[0,0,0,-1],[0,0,-1,0],[0,1,0,0],[1,0,0,0]]),
        "g2": np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]], dtype=complex),
        "g3": 1j*np.array([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]]),
        "g4": np.diag([1,1,-1,-1]).astype(complex),
        "g5": -np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=complex),
    }
    _need(isinstance(label, str), "Gamma label must be text")
    if re.fullmatch(r"sigma[1-4][1-4]", label):
        a, b = g["g"+label[-2]], g["g"+label[-1]]
        _need(label[-2] != label[-1], "Antisymmetric tensor needs distinct indices")
        return 0.5 * (a @ b - b @ a)
    parts = label.split(".")
    _need(parts and all(p in g for p in parts), f"Unsupported gamma label {label!r}")
    out = np.eye(4, dtype=complex)
    for p in parts:
        out = out @ g[p]
    return out


def _position(position, geometry):
    _need(len(position) == 4 and all(_integer(v) for v in position), "Source position must be [x,y,z,t]")
    _need(all(0 <= v < n for v, n in zip(position, (geometry.nx, geometry.ny, geometry.nz, geometry.nt))), "Source position outside lattice")
    return tuple(position)


def source_field(geometry: Geometry, *, kind, position, seed=None, grid_spacing=None):
    """Twelve spin/color-diluted RHS; shared spatial noise across those RHS.

    point has unit amplitude. z3-grid uses independent Z3 phases on a spatial
    periodic grid at the chosen time. Grid spacing divides each spatial extent.
    Grid of one site is supported. This is deterministic NumPy PCG64 noise, not
    a claim to reproduce the historical XQCD RNG stream.
    """
    x, y, z, t = _position(position, geometry)
    spatial = np.zeros((geometry.nz, geometry.ny, geometry.nx), dtype=complex)
    if kind == "point":
        _need(seed is None and grid_spacing is None, "Point source has no RNG/grid parameters")
        spatial[z, y, x] = 1
    elif kind == "z3-grid":
        _need(_integer(seed) and seed >= 0, "Z3 source requires an explicit nonnegative seed")
        _need(grid_spacing is not None and len(grid_spacing) == 3, "Z3 source requires [dx,dy,dz]")
        sizes = (geometry.nx, geometry.ny, geometry.nz)
        _need(all(_integer(d) and d > 0 and n % d == 0 for d, n in zip(grid_spacing, sizes)), "Grid spacing must divide the lattice")
        xs, ys, zs = [(start + np.arange(0, n, d)) % n for start, n, d in zip((x,y,z), sizes, grid_spacing)]
        coords = [(zz,yy,xx) for zz in zs for yy in ys for xx in xs]
        phases = np.exp(2j*np.pi*np.random.default_rng(seed).integers(0, 3, len(coords))/3)
        for coordinate, phase in zip(coords, phases):
            spatial[coordinate] = phase
    else:
        raise ValueError("Source kind must be point or z3-grid")
    source = np.zeros(geometry.propagator_shape, dtype=complex)
    for spin in range(4):
        for color in range(3):
            source[spin, color, spin, color, t] = spatial
    return NumericProduct("source", {"source": source}, {
        "profile": PROFILE, "geometry": asdict(geometry), "axes": PROP_AXES,
        "kind": kind, "position_xyzt": list(position), "seed": seed,
        "rng": "numpy.PCG64" if kind == "z3-grid" else None,
        "grid_spacing_xyz": grid_spacing, "normalization": "unit per nonzero site; spin/color dilution",
        "support_site_count":int(np.count_nonzero(spatial)),
        "sha256": array_sha256(source),
    })


def write_xqcd(path, array, geometry):
    """Exclusive write of >f8 [ss,sc,ri,ks,kc,t,z,y,x], no header."""
    a = _prop(array, geometry)
    with Path(path).open("xb") as stream:
        np.stack((a.real, a.imag), axis=2).astype(">f8").tofile(stream)
    return file_sha256(path)


def read_xqcd(path, geometry):
    expected = int(np.prod(geometry.propagator_shape)) * 16
    _need(Path(path).stat().st_size == expected, "XQCD output byte length disagrees with geometry")
    raw = np.fromfile(path, dtype=">f8").reshape((4,3,2,4,3)+geometry.shape)
    return _prop(raw[:,:,0] + 1j*raw[:,:,1], geometry)


def solve_reference(operator, source, geometry, *, tolerance=1e-11):
    """Solve explicit D X=source with a dense D, max dimension 4096.

    D acts on [sink_spin,sink_color,t,z,y,x] in C order. This verifies linear
    algebra and tensor plumbing, not Wilson/clover/overlap action physics.
    """
    _need(geometry.dimension <= 4096, "Dense reference backend is limited to dimension 4096")
    d = _finite(operator, "explicit Dirac matrix").astype(complex)
    s = _prop(source, geometry)
    _need(d.shape == (geometry.dimension, geometry.dimension), "Dirac matrix shape mismatch")
    _need(np.isfinite(tolerance) and 0 < tolerance < 1, "Invalid residual tolerance")
    rhs = s.reshape(12, -1).T
    solution = solve(d, rhs, assume_a="gen", check_finite=True)
    residual = np.linalg.norm(d @ solution - rhs, axis=0)
    norms = np.linalg.norm(rhs, axis=0)
    _need(np.all(norms > 0), "Every spin/color RHS must be nonzero")
    relative = residual / norms
    _need(np.all(relative <= tolerance), "Reference solve failed the true relative residual threshold")
    result = solution.T.reshape(geometry.propagator_shape)
    return NumericProduct("propagator", {"propagator": result}, {
        "backend": "explicit-matrix-reference", "scientific_status": "mathematical validation only; not a production action",
        "residual_definition": "||D*x-b||_2/||b||_2 per diluted RHS", "relative_residuals": relative.tolist(),
        "tolerance": tolerance, "operator_sha256": array_sha256(d), "source_sha256": array_sha256(s),
        "geometry": asdict(geometry), "axes": PROP_AXES, "sha256": array_sha256(result),
    })


def fourier_phase(geometry, momentum, source_position):
    _need(len(momentum) == 3 and all(_integer(v) for v in momentum), "Momentum must be integer [px,py,pz]")
    sx, sy, sz, _ = _position(source_position, geometry)
    z, y, x = np.indices((geometry.nz, geometry.ny, geometry.nx))
    px, py, pz = momentum
    return np.exp(-2j*np.pi*(px*(x-sx)/geometry.nx + py*(y-sy)/geometry.ny + pz*(z-sz)/geometry.nz))


def sequential_source(spectator, geometry, *, source_position, separation, initial_operator, momentum_initial):
    """CPU ST source: Gamma_initial G_spectator at (x0+T)%NT, phase -p_initial."""
    p = _prop(spectator, geometry)
    _need(_integer(separation) and 0 < separation < geometry.nt, "Require 0<T<NT")
    t = (_position(source_position, geometry)[3] + separation) % geometry.nt
    source = np.zeros_like(p)
    phase = fourier_phase(geometry, momentum_initial, source_position)
    source[:,:,:,:,t] = np.einsum("ab,ijbkzyx,zyx->ijakzyx", gamma(initial_operator), p[:,:,:,:,t], phase, optimize=True)
    return NumericProduct("sequential-source", {"source": source}, {
        "profile": PROFILE, "geometry": asdict(geometry), "axes": PROP_AXES,
        "spectator_sha256": array_sha256(p), "initial_operator": initial_operator,
        "momentum_initial": list(momentum_initial), "separation": separation,
        "source_position_xyzt": list(source_position), "insertion_lattice_time": t,
        "fourier_sign": -1, "sha256": array_sha256(source),
    })


def contract_two_point(first, second, geometry, *, source_position, source_operator, sink_operator, momentum):
    """CPU baseline: conj(first)*second, gamma5*Gamma_sink and Gamma_source*gamma5."""
    a, b = _prop(first, geometry), _prop(second, geometry)
    phase = fourier_phase(geometry, momentum, source_position)
    c = np.einsum("acbdtzyx,ecfdtzyx,bf,ea,zyx->t", a.conj(), b,
                  gamma("g5") @ gamma(sink_operator), gamma(source_operator) @ gamma("g5"), phase, optimize=True)
    t0 = _position(source_position, geometry)[3]
    return c[(t0 + np.arange(geometry.nt)) % geometry.nt]


def contract_three_point(final_propagator, sequential_propagator, geometry, *, source_position,
                         separation, current, final_operator, momentum_initial, momentum_final):
    """CPU ST baseline; stored j maps to raw lattice (x0+T-j)%NT.

    Sequential propagator solves mass m_initial on the source built from the
    spectator. The other argument has mass m_final. The output j runs from
    the initial insertion at x0+T towards the final insertion at x0.
    """
    a, b = _prop(final_propagator, geometry), _prop(sequential_propagator, geometry)
    _need(_integer(separation) and 0 < separation < geometry.nt, "Require 0<T<NT")
    fourier_phase(geometry, momentum_initial, source_position)
    fourier_phase(geometry, momentum_final, source_position)
    phase = fourier_phase(geometry, [p-q for p,q in zip(momentum_initial,momentum_final)], source_position)
    c = np.einsum("acbdtzyx,ecfdtzyx,bf,ea,zyx->t", a.conj(), b,
                  gamma("g5") @ gamma(current), gamma(final_operator) @ gamma("g5"), phase, optimize=True)
    t0 = _position(source_position, geometry)[3]
    return c[(t0 + separation - np.arange(geometry.nt)) % geometry.nt]


def _token(path):
    path = str(Path(path).resolve())
    _need(not any(c.isspace() for c in path) and "%" not in path and len(path) < 350,
          "XQCD token paths must have no whitespace/% and be shorter than 350 characters")
    return path


def _parse_input(text):
    fields = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, *value = line.split(maxsplit=1)
            _need(key not in fields and value, f"Duplicate or empty XQCD input key {key}")
            fields[key] = value[0]
    return fields


def solve_xqcd(source, geometry, *, mass, config_id, gauge, backend, workdir):
    """Run the confirmed overlap_inverter_cuda stdin/read_source contract.

    backend requires argv (already configured MPI environment), input_template,
    action_evidence, residual_pattern (one capture), residual_definition,
    residual_evidence, residual_tolerance, expected_residual_count. No guessed
    residual definition or action defaults. Cache inputs are copied to workdir
    before execution; templates must put writable cache paths there.
    """
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=False)
    _need(backend.get("profile") == "xqcd.overlap-inverter-stdin.v1", "Explicit supported backend profile required")
    argv = backend.get("argv")
    _need(isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv), "Backend argv must be an explicit nonempty string list")
    _need(Path(argv[0]).is_file(), "Backend executable/launcher must be an existing explicit path")
    for key in ("action_evidence", "residual_definition", "residual_evidence"):
        _need(isinstance(backend.get(key), str) and backend[key].strip(), f"Missing backend {key}")
    _need(_integer(config_id) and config_id >= 0, "XQCD configuration number must be a nonnegative integer")
    mass = float(mass)
    _need(np.isfinite(mass) and mass > 0 and float(f"{mass:.6f}") == mass, "XQCD profile requires positive masses exactly representable with six decimal places")
    gauge = Path(gauge).resolve()
    _need(gauge.is_file(), "A real gauge file is required for the XQCD backend")
    dependency_hashes = {"gauge": file_sha256(gauge), "launcher": file_sha256(argv[0])}
    for index, argument in enumerate(argv[1:],1):
        if Path(argument).is_file():
            dependency_hashes[f"argv_file_{index}"] = file_sha256(argument)
    for name, path in backend.get("dependency_files", {}).items():
        dest = (work / name).resolve()
        _need(work in dest.parents and not dest.exists(), "Dependency destination must be a new path under workdir")
        source_path = Path(path).resolve()
        _need(source_path.is_file(), f"Missing backend dependency {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("rb") as inp, dest.open("xb") as out:
            shutil.copyfileobj(inp, out)
        dependency_hashes[name] = file_sha256(dest)
    source_path = work / "source.bin"
    source_hash = write_xqcd(source_path, source, geometry)
    template_path = Path(backend["input_template"])
    dependency_hashes["input_template"] = file_sha256(template_path)
    mapping = dict(workdir=_token(work), source=_token(source_path), gauge=_token(gauge),
                   config_id=config_id, mass=f"{mass:.6f}", **asdict(geometry))
    text = Template(template_path.read_text(encoding="utf-8")).substitute(mapping)
    fields = _parse_input(text)
    _need({key for key in fields if key.startswith("masses:")} == {"masses:length", "masses:0"},
          "Single-mass backend input requires exactly length and index 0; extra or missing indices are invalid")
    _need(not any("REQUIRED" in value for value in fields.values()), "Complete every REQUIRED backend action value before execution")
    required = {"nx":str(geometry.nx),"ny":str(geometry.ny),"nz":str(geometry.nz),"nt":str(geometry.nt),
                "cfg_start":str(config_id),"cfg_end":str(config_id), "nsource":"1", "source_type00":"read_source",
                "source_text00":_token(source_path), "masses:length":"1", "masses:0":f"{mass:.6f}",
                "reload_kentucky_be":_token(gauge), "save_propagator":"1", "rotate_zt":"0"}
    _need(all(fields.get(k) == v for k,v in required.items()), "XQCD template contradicts geometry/source/mass/gauge/config/output profile")
    for key in ("kappa", "boundary_conditions", "cg_error", "maxiter", "poly_error", "one_minus_halfD", "Hwnoeigen", "ov_noeigen", "put_bak_low"):
        _need(key in fields, f"Missing explicit production action/solver input: {key}")
    _need(len(fields["boundary_conditions"].split())==4 and
          all(v in {"1","-1"} for v in fields["boundary_conditions"].split()),"This backend requires four explicit +/-1 boundary signs")
    for key in ("kappa","cg_error","poly_error"):
        value=float(fields[key])
        _need(np.isfinite(value) and value>0,f"Invalid positive backend input {key}")
    _need(int(fields["maxiter"])>0,"maxiter must be a positive integer")
    for key in ("prop_name", "Hwilson_name", "ov_evecname"):
        _need(key in fields, f"Missing scoped backend path {key}")
        p = Path(fields[key]).resolve()
        _need(work in p.parents, f"Writable backend path {key} must remain in new workdir")
        p.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(backend["residual_pattern"])
    _need(pattern.groups == 1, "Residual regex must have one numeric capture group")
    expected = backend.get("expected_residual_count")
    tolerance = backend.get("residual_tolerance")
    _need(_integer(expected) and expected >= 12 and np.isfinite(tolerance) and 0 < tolerance < 1,
          "Explicit residual count >=12 and positive tolerance required")
    input_path, log_path = work / "input.txt", work / "solver.log"
    with input_path.open("x", encoding="utf-8") as output:
        output.write(text)
    with input_path.open("rb") as stdin, log_path.open("xb") as stdout:
        completed = subprocess.run(argv, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT,
                                   cwd=work, shell=False, check=False, timeout=backend.get("timeout_seconds"))
    _need(completed.returncode == 0, f"XQCD exited {completed.returncode}; preserve {log_path}")
    residuals = [float(value) for value in pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))]
    _need(len(residuals) == expected and np.all(np.isfinite(residuals)) and np.all(np.asarray(residuals) >= 0)
          and np.all(np.asarray(residuals) <= tolerance), "Backend residual evidence missing, incomplete or above threshold")
    output_path = Path(str(source_path) + f".read00.m{mass:.6f}")
    result = read_xqcd(output_path, geometry)
    return NumericProduct("propagator", {"propagator":result}, {
        "backend": backend["profile"], "argv":argv, "action_evidence":backend["action_evidence"],
        "scientific_status":"external action; residual interpretation supplied by explicit backend contract",
        "mass":mass, "config_id":config_id, "geometry":asdict(geometry), "axes":PROP_AXES,
        "source_file_sha256":source_hash, "source_sha256":array_sha256(source),
        "dependencies":dependency_hashes, "input_sha256":file_sha256(input_path), "log_sha256":file_sha256(log_path),
        "output_file":str(output_path), "output_file_sha256":file_sha256(output_path), "sha256":array_sha256(result),
        "residuals":residuals, "residual_definition":backend["residual_definition"],
        "residual_evidence":backend["residual_evidence"], "residual_tolerance":tolerance,
    })


def save_product(product: NumericProduct, output_dir):
    """One numeric NPZ plus one JSON metadata file; never overwrite or clean up."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, value in product.arrays.items():
        _need(isinstance(name,str) and re.fullmatch(r"[A-Za-z0-9_.-]+",name), "Invalid numeric array name")
        _finite(value, name)
    arrays_path = path / "arrays.npz"
    with arrays_path.open("xb") as stream:
        np.savez(stream, **product.arrays)
    record = {"version":"lqcd.numeric-product.v1", "kind":product.kind, "metadata":product.metadata,
              "arrays_sha256":file_sha256(arrays_path), "arrays":{k:{"shape":list(v.shape),"dtype":v.dtype.str,
              "sha256":array_sha256(v)} for k,v in product.arrays.items()}}
    with (path/"product.json").open("x",encoding="utf-8") as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return record


def load_product(directory):
    path = Path(directory)
    record = json.loads((path/"product.json").read_text(encoding="utf-8"))
    _need(record.get("version") == "lqcd.numeric-product.v1", "Unsupported numeric product")
    _need(file_sha256(path/"arrays.npz") == record["arrays_sha256"], "Numeric product hash mismatch")
    with np.load(path/"arrays.npz",allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in archive.files}
    _need(set(arrays)==set(record["arrays"]),"Numeric product membership mismatch")
    for k,a in arrays.items():
        _finite(a,k)
        _need(array_sha256(a)==record["arrays"][k]["sha256"],"Numeric array metadata/hash mismatch")
    return NumericProduct(record["kind"], arrays, record["metadata"])


def run_generation(config, output_dir):
    """One explicit physical configuration. Stages: source, solve, contract, full.

    solve uses source-product input; contract uses a complete solved product.
    full closes primary solves, sequential source/solve and both contractions.
    No gauge/solver is silently replaced by the mathematical reference backend.
    """
    if isinstance(config,(str,Path)):
        config=json.loads(Path(config).read_text(encoding="utf-8"))
    _need(config.get("version")=="lqcd.generation.v1", "Unsupported generation configuration")
    _need(config.get("template") is not True,"Complete the generation template and set template=false before running")
    stage=config.get("stage")
    _need(stage in {"source","solve","contract","full"},"Choose source/solve/contract/full")
    _need(config.get("writer_profile")==PROFILE,"Choose the explicit CPU-ST writer profile")
    for k in ("ensemble","config_id","source_id"):
        _need(isinstance(config.get(k),str) and config[k],f"Explicit {k} required")
    g=Geometry(**config["geometry"])
    tensor_bytes=int(np.prod(g.propagator_shape))*16
    # Conservative live arrays, contraction workspace, raw I/O and NPZ staging.
    # Dense solve workspace is budgeted separately and still capped at dim 4096.
    memory_estimate=tensor_bytes*(4 if stage=="source" else 16)
    if config.get("backend",{}).get("profile")=="explicit-matrix-reference":
        memory_estimate += 6*g.dimension*g.dimension*16
    budget=config.get("max_memory_bytes",1024**3)
    _need(_integer(budget) and budget>0 and memory_estimate<=budget,
          f"CPU reference memory estimate {memory_estimate} bytes exceeds budget {budget}; use a smaller lattice or an explicit compute-node budget")
    source_spec=config["source"]
    position=source_spec["position"]
    _position(position,g)
    _need(config["source_id"]==f"t{position[3]}","v1 source_id must be t{x0}; extra noise replicas require a separate versioned identity")
    if stage!="source":
        physics=config["channel"]
        for key in ("initial_operator","final_operator","current"):
            gamma(physics[key])
        for key in ("momentum_initial","momentum_final"):
            fourier_phase(g,physics[key],position)
        _need(_integer(physics["separation"]) and 0<physics["separation"]<g.nt,"Require physical 0<T<NT")
        if source_spec["kind"]=="z3-grid":
            _need(config.get("grid_estimator")=="zero-momentum-source-average",
                  "Grid contractions require explicit grid_estimator=zero-momentum-source-average")
            _need(physics["momentum_initial"]==physics["momentum_final"]==[0,0,0],
                  "This CPU reference grid estimator supports zero momentum only; use a point source for nonzero momentum")
    out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    stages={}
    if stage in {"source","full"}:
        src=source_field(g,**source_spec)
        arrays={"source":src.arrays["source"]}
        stages["source"]=src.metadata
    else:
        previous=load_product(config["input_product"])
        _need(previous.metadata.get("geometry")==asdict(g),"Input product geometry mismatch")
        for k in ("ensemble","config_id","source_id"):
            _need(previous.metadata.get(k)==config[k],f"Input product {k} mismatch")
        _need(previous.metadata.get("source")==source_spec,"Input source specification mismatch")
        arrays=dict(previous.arrays)
        stages=dict(previous.metadata["stages"])
    if stage in {"solve","full"}:
        masses=config["masses"]
        _need(set(masses)=={"initial","final","spectator"} and all(np.isfinite(float(v)) and float(v)>0 for v in masses.values()),"Specify initial/final active and spectator masses")
        backend=config["backend"]
        def inverter(source,name,mass_role):
            if backend.get("profile")=="explicit-matrix-reference":
                _need(config.get("scientific_mode")=="mathematical-validation","Explicit matrices require mathematical-validation mode")
                operator_path=backend["operators"][mass_role]
                operator_hash=file_sha256(operator_path)
                operator=np.load(operator_path,allow_pickle=False)
                result=solve_reference(operator,source,g,tolerance=backend["residual_tolerance"])
                result.metadata["operator_file_sha256"]=operator_hash
                result.metadata["mass_label"]=masses[mass_role]
                return result
            _need(config.get("scientific_mode")=="external-production-action","External backend requires external-production-action mode")
            return solve_xqcd(source,g,mass=masses[mass_role],config_id=int(config["config_id"]),
                              gauge=config["gauge"],backend=backend,workdir=out/name)
        for role in ("initial","final","spectator"):
            product=inverter(arrays["source"],"solve_"+role,role)
            arrays["prop_"+role]=product.arrays["propagator"]
            stages["solve_"+role]=product.metadata
        physics=config["channel"]
        sequential=sequential_source(arrays["prop_spectator"],g,source_position=position,
                                     separation=physics["separation"],initial_operator=physics["initial_operator"],
                                     momentum_initial=physics["momentum_initial"])
        arrays["sequential_source"]=sequential.arrays["source"]
        stages["sequential_source"]=sequential.metadata
        seqprop=inverter(arrays["sequential_source"],"solve_sequential","initial")
        arrays["prop_sequential"]=seqprop.arrays["propagator"]
        stages["solve_sequential"]=seqprop.metadata
    if stage in {"contract","full"}:
        physics=config["channel"]
        if stage=="contract":
            _need(previous.metadata.get("channel")==physics and previous.metadata.get("masses")==config["masses"],"Contraction channel/masses disagree with solved product")
        for role in ("initial","final"):
            op=physics[role+"_operator"]
            arrays["c2_"+role]=contract_two_point(arrays["prop_spectator"],arrays["prop_"+role],g,
                source_position=position,source_operator=op,sink_operator=op,momentum=physics["momentum_"+role])
        arrays["c3"]=contract_three_point(arrays["prop_final"],arrays["prop_sequential"],g,source_position=position,
             separation=physics["separation"],current=physics["current"],final_operator=physics["final_operator"],
             momentum_initial=physics["momentum_initial"],momentum_final=physics["momentum_final"])
        support=stages["source"]["support_site_count"]
        for name in ("c2_initial","c2_final","c3"):
            arrays[name]=arrays[name]/support
        stages["contraction"]={"profile":PROFILE,"fourier_sign":-1,"c2_time":"source-forward",
             "c3_time":"sink-backward", "c3_physical_time":"j from initial at x0+T to final at x0; decreasing lattice time",
             "c3_mass_order":["initial","final","spectator"],"c2_mass_order":["spectator","active"],
             "source_average_divisor":support,"grid_estimator":config.get("grid_estimator"),
             "input_sha256":{k:array_sha256(v) for k,v in arrays.items() if k.startswith("prop_")}}
    metadata={k:config.get(k) for k in ("ensemble","config_id","source_id","scientific_mode","masses","channel","source")}
    metadata.update(profile=PROFILE,geometry=asdict(g),stage=stage,stages=stages,
                    backend_status="explicit-operator-reference" if config.get("scientific_mode")=="mathematical-validation" else "external-numerics-unverified",
                    estimated_memory_bytes=memory_estimate, memory_budget_bytes=budget,
                    artifacts={"arrays":str(out/"arrays.npz"),"metadata":str(out/"product.json")},
                    propagator_axes=PROP_AXES,correlator_axes=["time"],
                    config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                    code_sha256=file_sha256(__file__))
    product=NumericProduct("generation-"+stage,arrays,metadata)
    save_product(product,out)
    return product


def correlator_records(product: NumericProduct):
    """Return three schema-independent records for the parent's SQLite writer.

    Each record is {path, arrays, metadata}; arrays contain (2,NT) [real,imag].
    No measurements are duplicated or written as individual files here. Bare
    complex contractions are retained; analysis must declare its projection.
    """
    m=product.metadata
    _need(all(k in product.arrays for k in ("c2_initial","c2_final","c3")),"Product has no complete contractions")
    g=Geometry(**m["geometry"])
    channel=m["channel"]
    x0=m["source"]["position"][3]
    masses={k:f"{float(v):.6f}" for k,v in m["masses"].items()}
    _need(all(float(masses[k])==float(v) for k,v in m["masses"].items()),"Output basename requires six-decimal masses without rounding")
    number=m["config_id"]
    _need(re.fullmatch(r"[^_/\\]+",number) is not None,"Configuration ID cannot contain path/underscore separators")
    def make(kind,key,path,identity,time):
        a=product.arrays[key]
        metadata=dict(identity,kind=kind,ensemble=m["ensemble"],config_id=number,source_id=m["source_id"],
            source_time=x0,nt=g.nt,writer_profile=PROFILE,time_convention=time,storage="split-ri",
            component="complex",profile="savedata.pickle.numeric.v1",backend_status=m["backend_status"],
            scientific_mode=m["scientific_mode"],physics_status="unconfirmed",
            convention_evidence="CPU ST reference contraction; see generation profile and preserved dependencies",
            generation_config_sha256=m["config_sha256"],dependencies=m["stages"],source_position_xyzt=m["source"]["position"])
        array_key=identity.get("current",identity["sink_operator"]+"_"+identity["source_operator"])
        return {"path":path,"arrays":{array_key:np.stack([a.real,a.imag])},"metadata":metadata}
    records=[]
    for role in ("initial","final"):
        op=channel[role+"_operator"]
        mom=channel["momentum_"+role]
        token="".join(map(str,mom))
        path=f"2pt_{number}_t{x0}_m{masses['spectator']}_m{masses[role]}_mom{token}.bin"
        identity=dict(ordered_masses=[masses["spectator"],masses[role]],source_operator=op,sink_operator=op,
                      momentum_initial=mom,momentum_final=mom)
        # Paths contain a role directory so equal physical channels can still be
        # stored as separately labeled products; analysis refuses duplicate C2s.
        records.append(make("2pt","c2_"+role,role+"/"+path,identity,"source-forward"))
    pi,pf=channel["momentum_initial"],channel["momentum_final"]
    gi,gf=channel["initial_operator"],channel["final_operator"]
    T=channel["separation"]
    path=f"3pt_{number}_t{x0}_T{T}_m{masses['initial']}_m{masses['final']}_{gf}_m{masses['spectator']}_{gi}_mom{''.join(map(str,pi))}_{''.join(map(str,pf))}.bin"
    identity=dict(ordered_masses=[masses[k] for k in ("initial","final","spectator")],
        source_operator=gi,sink_operator=gf,momentum_initial=pi,momentum_final=pf,separation=T,current=channel["current"],
        raw_lattice_mapping="(source_time+separation-j)%nt; j=0 initial, j=T final")
    records.append(make("3pt","c3",path,identity,"initial-forward"))
    return records
