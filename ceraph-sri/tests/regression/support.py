"""Helpers for published-number regression tests."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"

# Closed forms are compared exactly. Most simulated values stay inside 1e-6.
# v2's weighted layout coefficient moves by about 1e-5 relative between BLAS runs.
SIM_RTOL = 1e-6
SIM_ATOL = 1e-12
V2_FIT_RTOL = 5e-5


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(name: str) -> dict:
    return load_json(SNAPSHOTS / name)


def load_suite(version: str):
    """Load experiments/unveiling/<version>/suite.py under a unique module name."""
    path = ROOT / "experiments" / "unveiling" / version / "suite.py"
    module_name = f"sri_regression_suite_{version}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def quiet():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        yield buffer


def assert_close(got, expected, *, rtol: float, atol: float, path: str = "value") -> None:
    """Compare numbers, dicts, and lists. Fail with the relative delta."""
    if isinstance(expected, dict):
        assert isinstance(got, dict), path
        missing = set(expected) - set(got)
        assert not missing, f"{path} missing {sorted(missing)}"
        for key, value in expected.items():
            assert_close(got[key], value, rtol=rtol, atol=atol, path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected), path
        for index, value in enumerate(expected):
            assert_close(got[index], value, rtol=rtol, atol=atol, path=f"{path}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert got == expected, f"{path}: {got!r} != {expected!r}"
        return
    if isinstance(expected, int) and not isinstance(got, float):
        assert got == expected, f"{path}: {got!r} != {expected!r}"
        return
    got_f = float(got)
    exp_f = float(expected)
    delta = got_f - exp_f
    limit = atol + rtol * abs(exp_f)
    assert abs(delta) <= limit, (
        f"{path}: got {got_f!r} expected {exp_f!r} delta {delta!r} "
        f"rel {abs(delta) / max(abs(exp_f), 1e-30)!r} limit {limit!r}"
    )
