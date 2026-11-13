"""A physical configuration drives source -> solve -> contract -> DB -> fit.

The CPU reference producer and existing XQCD backend are explicit profiles.
Checking a configuration never starts a solver. Running requires a new output
directory and retains all intermediate products and provenance.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from dataclasses import asdict
from string import Template

from .generation import Geometry, PROFILE, gamma, run_generation, correlator_records, _token, _parse_input
from .writer import create_generated_store


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _path(base, value):
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_xqcd_template(backend, geometry, mass, config_id, gauge, work):
    """Validate all task-specific substitutions before any solver is started."""
    source = work / "source.bin"
    text = Template(Path(backend["input_template"]).read_text(encoding="utf-8")).substitute(
        workdir=_token(work), source=_token(source), gauge=_token(gauge), config_id=int(config_id),
        mass=f"{float(mass):.6f}", **asdict(geometry))
    fields = _parse_input(text)
    _require({key for key in fields if key.startswith("masses:")} == {"masses:length", "masses:0"},
             "Single-mass backend input requires exactly one mass at index 0")
    _require(not any("REQUIRED" in v for v in fields.values()), "Complete the backend action template")
    required = {"nx": str(geometry.nx), "ny": str(geometry.ny), "nz": str(geometry.nz), "nt": str(geometry.nt),
                "cfg_start": str(int(config_id)), "cfg_end": str(int(config_id)), "nsource": "1",
                "source_type00": "read_source", "source_text00": _token(source), "masses:length": "1",
                "masses:0": f"{float(mass):.6f}", "reload_kentucky_be": _token(gauge),
                "save_propagator": "1", "rotate_zt": "0"}
    _require(all(fields.get(k) == v for k, v in required.items()), "Backend input contradicts the physical task")
    for key in ("kappa", "boundary_conditions", "cg_error", "maxiter", "poly_error", "one_minus_halfD", "Hwnoeigen", "ov_noeigen", "put_bak_low"):
        _require(key in fields, f"Missing explicit backend action field: {key}")
    for key in ("prop_name", "Hwilson_name", "ov_evecname"):
        _require(key in fields and work in Path(fields[key]).resolve().parents, f"Writable backend {key} must stay in its new work directory")
    for name in backend.get("dependency_files", {}):
        _require(work in (work / name).resolve().parents, "Backend dependency destination escapes the work directory")


def plan_workflow(config_path):
    """Validate physical labels and paths without creating files or starting work."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _require(config.get("version") == "lqcd.easy.v1", "Unsupported easy-workflow configuration")
    _require(config.get("template") is not True, "Fill the physical configuration, then change template to false")
    _require(isinstance(config.get("ensemble"), str) and config["ensemble"].strip(), "Name the ensemble explicitly")
    base = config_path.parent
    output = Path(_path(base, config["output"]))
    _require(not output.exists(), "Choose a new output directory outside the excluded data")
    geometry = Geometry(**config["geometry"])
    process = config["process"]
    masses = {}
    for role in ("initial", "final", "spectator"):
        label = process[role + "_quark"]
        _require(isinstance(label, str) and label in config["quark_masses"], f"Unknown {role} quark label")
        mass = config["quark_masses"][label]
        try:
            _require(isinstance(mass, str) and Decimal(mass).is_finite() and Decimal(mass) > 0 and
                     Decimal(mass) == Decimal(f"{float(mass):.6f}"), "Quark masses must be positive decimal strings with six-decimal writer compatibility")
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid mass for {label}") from exc
        masses[role] = mass
    for operator in ("initial_operator", "final_operator", "current"):
        gamma(process[operator])
    for name in ("momentum_initial", "momentum_final"):
        vector = process[name]
        _require(isinstance(vector, list) and len(vector) == 3 and all(_integer(x) for x in vector), "Momenta are integer [nx,ny,nz] triples")
    _require(_integer(process.get("separation")) and 0 < process["separation"] < geometry.nt, "Choose 0 < T < NT")
    source = config["sources"]
    _require(source.get("kind") in {"point", "z3-grid"}, "Choose point or z3-grid sources")
    positions = source.get("positions")
    _require(isinstance(positions, list) and positions, "Provide at least one [x,y,z,t] source position")
    source_times = []
    for position in positions:
        _require(isinstance(position, list) and len(position) == 4 and all(_integer(x) for x in position), "Source positions are integer [x,y,z,t]")
        _require(all(0 <= x < size for x, size in zip(position, (geometry.nx, geometry.ny, geometry.nz, geometry.nt))), "Source position outside the lattice")
        source_times.append(position[3])
    _require(len(source_times) == len(set(source_times)), "v1 supports one source measurement per configuration and source time")
    if source["kind"] == "z3-grid":
        _require(_integer(source.get("base_seed")) and source["base_seed"] >= 0, "Choose a nonnegative Z3 base_seed")
        spacing = source.get("grid_spacing")
        _require(isinstance(spacing, list) and len(spacing) == 3 and
                 all(_integer(d) and d > 0 and n % d == 0 for d, n in zip(spacing, (geometry.nx, geometry.ny, geometry.nz))), "Z3 grid spacing must divide each spatial extent")
        _require(config.get("grid_estimator") == "zero-momentum-source-average", "Z3 contraction requires the explicit zero-momentum source-average estimator")
        _require(process["momentum_initial"] == process["momentum_final"] == [0, 0, 0], "Use point sources for nonzero momentum with this reference profile")
    backend_base = base
    backend = copy.deepcopy(config["backend"])
    if isinstance(backend, str):
        profile_path = Path(_path(base, backend))
        backend_base = profile_path.parent
        backend = json.loads(profile_path.read_text(encoding="utf-8"))
    _require(isinstance(backend, dict) and backend.get("template") is not True, "Complete and validate the backend profile once before using it")
    profile = backend.get("profile")
    if profile == "explicit-matrix-reference":
        _require(config.get("scientific_mode") == "mathematical-validation", "Explicit matrices are a mathematical validation backend")
        backend["operators"] = {role: _path(backend_base, value) for role, value in backend["operators"].items()}
        _require(set(backend["operators"]) == {"initial", "final", "spectator"}, "Provide all three explicit reference operators")
        _require(all(Path(p).is_file() for p in backend["operators"].values()), "An explicit operator file is missing")
        _require(geometry.dimension <= 4096, "Reference matrix solver supports dimension <= 4096")
        import numpy as np
        for operator_path in backend["operators"].values():
            matrix = np.load(operator_path, mmap_mode="r", allow_pickle=False)
            _require(isinstance(matrix, np.ndarray) and matrix.shape == (geometry.dimension, geometry.dimension)
                     and matrix.dtype.kind in "fciu" and np.all(np.isfinite(matrix)), "Reference operator shape/type/finiteness disagrees with geometry")
    elif profile == "xqcd.overlap-inverter-stdin.v1":
        _require(config.get("scientific_mode") == "external-production-action", "Choose external-production-action explicitly for XQCD")
        for key in ("input_template",):
            backend[key] = _path(backend_base, backend[key])
            _require(Path(backend[key]).is_file(), f"Missing {key}")
        argv = backend.get("argv")
        _require(isinstance(argv, list) and argv and all(isinstance(x, str) and x for x in argv), "Provide the explicit solver/launcher argument list")
        backend["argv"][0] = _path(backend_base, argv[0])
        _require(Path(backend["argv"][0]).is_file(), "Solver executable or MPI launcher is missing")
        backend["dependency_files"] = {k: _path(backend_base, v) for k, v in backend.get("dependency_files", {}).items()}
        _require(all(Path(v).is_file() for v in backend["dependency_files"].values()), "A solver dependency file is missing")
        for key in ("action_evidence", "residual_pattern", "residual_definition", "residual_evidence"):
            _require(isinstance(backend.get(key), str) and backend[key].strip(), f"Provide backend {key}")
        try:
            pattern = re.compile(backend["residual_pattern"])
        except re.error as exc:
            raise ValueError("Invalid residual regular expression") from exc
        _require(pattern.groups == 1 and _integer(backend.get("expected_residual_count")) and backend["expected_residual_count"] >= 12,
                 "Residual evidence needs one numeric capture and an expected count of at least 12")
        if backend.get("timeout_seconds") is not None:
            _require(isinstance(backend["timeout_seconds"], (int, float)) and math.isfinite(backend["timeout_seconds"]) and backend["timeout_seconds"] > 0,
                     "Backend timeout must be positive")
    else:
        raise ValueError("Choose an explicit supported solver profile")
    tolerance = backend.get("residual_tolerance")
    _require(isinstance(tolerance, (int, float)) and 0 < tolerance < 1, "Choose a solver residual threshold")
    configurations = config.get("configurations")
    _require(isinstance(configurations, list) and configurations, "Provide configuration IDs and gauge paths")
    seen_configs, tasks = set(), []
    for item in configurations:
        config_id = item.get("id")
        _require(isinstance(config_id, str) and re.fullmatch(r"[0-9]+", config_id), "Configuration IDs must be digit strings")
        _require(config_id == str(int(config_id)), "Use canonical decimal configuration IDs without leading zeros")
        _require(config_id not in seen_configs, "Duplicate configuration ID")
        seen_configs.add(config_id)
        gauge = None
        if profile == "xqcd.overlap-inverter-stdin.v1":
            gauge = _path(base, item["gauge"])
            _require(Path(gauge).is_file(), "Gauge path is missing or excluded")
        for position in positions:
            source_spec = {"kind": source["kind"], "position": position}
            if source["kind"] == "z3-grid":
                token = f"{source['base_seed']}:{config_id}:{position}"
                source_spec.update(seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little"),
                                   grid_spacing=source["grid_spacing"])
            generation = {"version": "lqcd.generation.v1", "stage": "full", "writer_profile": PROFILE,
                          "ensemble": config["ensemble"], "config_id": config_id, "source_id": f"t{position[3]}",
                          "geometry": config["geometry"], "source": source_spec, "masses": masses,
                          "channel": {k: process[k] for k in ("initial_operator", "final_operator", "current", "momentum_initial", "momentum_final", "separation")},
                          "backend": backend, "gauge": gauge, "scientific_mode": config["scientific_mode"],
                          "max_memory_bytes": config.get("max_memory_bytes", 1024**3)}
            if source["kind"] == "z3-grid":
                generation["grid_estimator"] = config["grid_estimator"]
            if profile == "xqcd.overlap-inverter-stdin.v1":
                for role in ("initial", "final", "spectator", "sequential"):
                    mass = masses["initial" if role == "sequential" else role]
                    work = output / "products" / f"cfg{config_id}-t{position[3]}" / ("solve_" + role)
                    _check_xqcd_template(backend, geometry, mass, config_id, gauge, work)
            # Same conservative pre-allocation budget as the producer.
            estimate = 16 * int(math.prod(geometry.propagator_shape)) * 16
            if profile == "explicit-matrix-reference":
                estimate += 6 * geometry.dimension**2 * 16
            _require(estimate <= generation["max_memory_bytes"], f"Estimated {estimate} bytes exceed the chosen memory budget; choose a compute-node budget explicitly")
            tasks.append({"name": f"cfg{config_id}-t{position[3]}", "config": generation, "estimated_memory_bytes": estimate})
    analysis = config.get("analysis")
    if analysis is not None:
        _require(isinstance(analysis, dict) and analysis.get("type") in {"two-point", "joint-c2-c3"}, "Choose an explicit supported analysis type")
        settings = analysis.get("settings", {})
        _require(isinstance(settings, dict) and settings.get("template") is not True, "Complete the analysis settings")
        if settings.get("max_nfev") is not None:
            _require(_integer(settings["max_nfev"]) and settings["max_nfev"] > 0, "max_nfev must be a positive integer")
        bin_size = settings.get("bin_size")
        _require(_integer(bin_size) and bin_size > 0 and len(configurations) % bin_size == 0 and len(configurations) // bin_size >= 2,
                 "Analysis requires at least two equal-size configuration bins, without a remainder")
        _require(isinstance(settings.get("svd_rcond"), (float, int)) and 0 < settings["svd_rcond"] < 1, "Choose an explicit covariance SVD cutoff")
        _require(settings.get("spectral_assumption") == "positive-diagonal-periodic", "Choose the diagonal C2 spectral assumption explicitly")
        from .joint_analysis import _sector
        if analysis["type"] == "two-point":
            _require(settings.get("version") == "lqcd.fit2.v1" and analysis.get("role") in {"initial", "final"}, "Invalid two-point fit version/role")
            _require(isinstance(analysis.get("real_projection_evidence"), str) and analysis["real_projection_evidence"].strip(), "Justify the real positive C2 projection")
            count, *_ = _sector(settings["model"])
            windows = {"two_point": settings.get("window")}
            parameter_count = 2 * count
        else:
            _require(settings.get("version") == "lqcd.joint-fit.v1" and settings.get("three_point_spectral_assumption") == "forward-only", "Invalid joint fit version/spectral assumption")
            _require(settings.get("c3_time_mapping") == "initial-forward", "Newly generated C3 uses initial-forward physical time")
            _require(settings.get("alignment") == "exact", "Generated joint channels use the same exact configuration population")
            _require((masses["initial"], process["initial_operator"], process["momentum_initial"]) !=
                     (masses["final"], process["final_operator"], process["momentum_final"]),
                     "Identical initial/final C2 needs a shared-sector analysis model")
            initial, *_ = _sector(settings["model_initial"])
            final, *_ = _sector(settings["model_final"])
            projection = settings.get("current_projection", {})
            phase = projection.get("phase", [])
            _require(isinstance(phase, list) and len(phase) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in phase)
                     and abs(sum(v*v for v in phase) - 1) < 1e-10, "Declare the current's unit complex phase")
            _require(isinstance(projection.get("scale"), (int, float)) and math.isfinite(projection["scale"]) and projection["scale"] != 0
                     and isinstance(projection.get("evidence"), str) and projection["evidence"].strip(), "Declare current normalization and physical evidence")
            import numpy as np
            matrix, bounds = np.asarray(settings.get("matrix_initial")), np.asarray(settings.get("matrix_bounds"))
            _require(matrix.shape == (final, initial) and bounds.shape == (final, initial, 2) and
                     matrix.dtype.kind in "fiu" and bounds.dtype.kind in "fiu" and np.all(np.isfinite(matrix)) and np.all(np.isfinite(bounds))
                     and np.all(bounds[:, :, 0] < matrix) and np.all(matrix < bounds[:, :, 1]), "Provide current matrix starting values and bounds")
            windows = settings.get("windows", {})
            _require(set(windows) == {"initial", "final", "three_point"}, "Set every joint fit window")
            parameter_count = 2 * initial + 2 * final + initial * final
        observations = 0
        for role, times in windows.items():
            upper = process["separation"] if role == "three_point" else geometry.nt
            lower = 1 if role == "three_point" else 0
            _require(isinstance(times, list) and times and all(_integer(v) and lower <= v < upper for v in times)
                     and all(b > a for a, b in zip(times, times[1:])), "Choose increasing fit times inside the physical interval; exclude C3 contacts")
            observations += len(times)
        _require(observations > parameter_count and len(configurations) // bin_size - 1 > parameter_count,
                 "There is insufficient independent information for the requested fit")
    return {"config": config, "output": str(output), "tasks": tasks,
            "configuration_path": str(config_path), "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}


def run_workflow(config_path):
    plan = plan_workflow(config_path)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "products").mkdir(exist_ok=False)
    generated_records, products = [], []
    for task in plan["tasks"]:
        product = run_generation(task["config"], output / "products" / task["name"])
        generated_records.extend(correlator_records(product))
        products.append({"name": task["name"], "artifacts": product.metadata["artifacts"],
                         "backend_status": product.metadata["backend_status"]})
    store = create_generated_store(output / "correlators", generated_records,
                                   {"workflow_version": "lqcd.easy.v1", "physical_configuration": plan["config"],
                                    "configuration_sha256": plan["configuration_sha256"], "products": products})
    analysis_request = plan["config"].get("analysis")
    analysis_result = None
    if analysis_request:
        selections = [json.loads(Path(p).read_text(encoding="utf-8")) for p in store["selections"]]
        role_selections = {}
        for role, prefix in (("initial", "initial/"), ("final", "final/"), ("three_point", "3pt_")):
            candidates = [s for s in selections if all(row["path"].startswith(prefix) for row in s["measurements"])]
            _require(len(candidates) == 1, f"Cannot select a unique generated {role} channel")
            role_selections[role] = candidates[0]
        settings = copy.deepcopy(analysis_request["settings"])
        settings["configuration_order"] = [row["id"] for row in plan["config"]["configurations"]]
        if analysis_request["type"] == "two-point":
            from .analysis import fit_two_point
            _require(analysis_request.get("role") in {"initial", "final"}, "Choose the two-point role")
            _require(isinstance(analysis_request.get("real_projection_evidence"), str) and analysis_request["real_projection_evidence"].strip(),
                     "Provide the physical justification for the real positive C2 projection")
            selected = role_selections[analysis_request["role"]]
            selected["component"] = "real"
            selected["convention_evidence"] += "; " + analysis_request["real_projection_evidence"]
            analysis_result = fit_two_point(store["database"], selected, settings, verify_database_hash=True)
        elif analysis_request["type"] == "joint-c2-c3":
            from .joint_analysis import fit_joint_c2_c3
            for role in ("initial", "final"):
                role_selections[role]["component"] = "real"
            analysis_result = fit_joint_c2_c3(store["database"], role_selections, settings, verify_database_hash=True)
        else:
            raise ValueError("Unknown analysis type")
        with (output / "analysis.json").open("x", encoding="utf-8") as stream:
            json.dump(analysis_result, stream, ensure_ascii=True, indent=2, allow_nan=False)
    result = {"version": "lqcd.workflow-result.v1", "database": store, "products": products,
              "analysis": str(output / "analysis.json") if analysis_result else None,
              "scientific_status": "CPU reference workflow; external backend and physical conventions require the recorded validation"}
    with (output / "workflow.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
    return result
