"""AWS Braket adapter: dry-run by default; live submit under hard cost caps.

Live path requires:
  BRAKET_SUBMIT=1
  plan.submit=True
  estimated_cost_usd <= plan.cost_cap_usd
  AWS credentials with Braket + S3 permissions

Without credentials, use emulate_quantum_task() for local plumbing tests.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"

DEVICE_ALIASES = {
    "sv1": SV1_ARN,
    "dm1": DM1_ARN,
}

# Conservative per-shot heuristics used as a pre-submit gate (not a billing API).
_COST_PER_SHOT_USD = {
    "sv1": 0.0,  # AWS often bills simulator via Braket task minimum; treat as ~$0.075/task
    "dm1": 0.0,
    "qpu": 0.03,
}
_SIMULATOR_TASK_MINIMUM_USD = 0.075


def resolve_device_arn(device: str | None = None, device_arn: str | None = None) -> str:
    """Map ``sv1`` / ``dm1`` aliases (or a raw ARN) to a Braket device ARN."""
    if device_arn:
        return device_arn
    if device is None:
        return SV1_ARN
    key = device.strip().lower()
    if key in DEVICE_ALIASES:
        return DEVICE_ALIASES[key]
    if key.startswith("arn:"):
        return device
    raise ValueError(f"unknown device alias or ARN: {device!r}")


@dataclass
class BraketJobPlan:
    """Device-agnostic job payload for a CV sensing analogue or qubit toy."""

    job_name: str
    device_arn: str
    shots: int
    task: str
    parameters: dict[str, Any] = field(default_factory=dict)
    cost_cap_usd: float = 5.0
    region: str = "us-east-1"
    submit: bool = False
    s3_bucket: str | None = None
    s3_prefix: str = "sri-braket"

    def validate(self) -> None:
        if not self.job_name or not self.device_arn:
            raise ValueError("job_name and device_arn are required")
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.task not in {
            "cv_sensing_analogue",
            "qubit_toy_ising",
            "hybrid_e1_e4",
            "sphota_yoga_buddhi",
        }:
            raise ValueError(f"unsupported task: {self.task}")
        if self.cost_cap_usd <= 0:
            raise ValueError("cost_cap_usd must be positive")


def build_smoke_plan(
    *,
    device_arn: str | None = None,
    device: str | None = None,
    shots: int = 1000,
    task: str = "cv_sensing_analogue",
    cost_cap_usd: float = 5.0,
    submit: bool = False,
    s3_bucket: str | None = None,
    region: str = "us-east-1",
    n_qubits: int = 4,
) -> BraketJobPlan:
    """First-job payload. Defaults to the SV1 state-vector simulator ARN."""
    resolved_arn = resolve_device_arn(device=device, device_arn=device_arn)
    parameters: dict[str, Any] = {
        "signal_variance": 1.0,
        "noise_variance": 1.0,
        "coupling": 1.0,
        "efficiency": 0.8,
        "reference_r": 0.7,
        "encoding": "gaussian_analogue_via_classical_postprocess",
        "note": (
            "CV homodyne sensing is simulated classically unless a photonic "
            "Braket device is selected; SV1/DM1 path validates job plumbing only."
        ),
    }
    if task == "qubit_toy_ising":
        parameters = {
            "n_qubits": int(n_qubits),
            "hamiltonian": "ising_chain",
            "j_coupling": 1.0,
            "device_alias": device or ("dm1" if "dm1" in resolved_arn else "sv1"),
            "note": (
                "Small qubit toy for account/device smoke tests. "
                "DM1 enables density-matrix / noise simulations; still not CV MMSE."
            ),
        }
    if task == "hybrid_e1_e4":
        parameters = {
            "seeds": [108, 109, 110],
            "shots_cv": 20000,
            "dimension": 4096,
            "campaign_bundle": ["advantage", "efficiency"],
            "note": "Hybrid Job runs classical CV E1–E4 on Braket-managed compute.",
        }
    return BraketJobPlan(
        job_name="sri-e1-braket-smoke",
        device_arn=resolved_arn,
        shots=shots,
        task=task,
        parameters=parameters,
        cost_cap_usd=cost_cap_usd,
        region=region,
        submit=submit,
        s3_bucket=s3_bucket or os.environ.get("BRAKET_S3_BUCKET"),
    )


def estimate_cost_usd(plan: BraketJobPlan) -> float:
    """Conservative pre-submit cost gate (not an AWS quote)."""
    arn = plan.device_arn.lower()
    if "quantum-simulator" in arn or ":sv1" in arn or ":dm1" in arn:
        return max(_SIMULATOR_TASK_MINIMUM_USD, plan.shots * _COST_PER_SHOT_USD["sv1"])
    return max(0.3, plan.shots * _COST_PER_SHOT_USD["qpu"])


def dry_run_validate(plan: BraketJobPlan) -> dict[str, Any]:
    """Validate schema and cost gate. Does not submit."""
    plan.validate()
    estimated = estimate_cost_usd(plan)
    submit_requested = os.environ.get("BRAKET_SUBMIT", "0") == "1"
    return {
        "status": "dry_run_ok",
        "submit_requested": submit_requested,
        "estimated_cost_usd": estimated,
        "within_cost_cap": estimated <= plan.cost_cap_usd,
        "plan": asdict(plan),
    }


def _assert_live_gates(plan: BraketJobPlan) -> float:
    plan.validate()
    if os.environ.get("BRAKET_SUBMIT", "0") != "1":
        raise RuntimeError("Set BRAKET_SUBMIT=1 to enable live Braket submission")
    if not plan.submit:
        raise RuntimeError("plan.submit must be True for live submission")
    estimated = estimate_cost_usd(plan)
    if estimated > plan.cost_cap_usd:
        raise RuntimeError(
            f"estimated cost ${estimated:.4f} exceeds cost_cap_usd ${plan.cost_cap_usd:.4f}"
        )
    if not plan.s3_bucket:
        raise RuntimeError("s3_bucket is required for live submit (set BRAKET_S3_BUCKET)")
    return estimated


def build_ising_circuit(n_qubits: int = 4, j_coupling: float = 1.0):
    """Build a small 1D Ising-prep circuit for SV1/QPU smoke tests."""
    try:
        from braket.circuits import Circuit
    except ImportError as error:
        raise RuntimeError(
            "amazon-braket-sdk is required for circuit construction"
        ) from error
    if n_qubits < 2 or n_qubits > 8:
        raise ValueError("n_qubits must be in [2, 8] for the smoke toy")
    circuit = Circuit()
    for qubit in range(n_qubits):
        circuit.h(qubit)
    for qubit in range(n_qubits - 1):
        circuit.zz(qubit, qubit + 1, j_coupling)
    return circuit


def emulate_quantum_task(plan: BraketJobPlan) -> dict[str, Any]:
    """Local classical stand-in for SV1 plumbing tests (no AWS spend)."""
    plan.validate()
    estimated = estimate_cost_usd(plan)
    n_qubits = int(plan.parameters.get("n_qubits", 4))
    # Deterministic pseudo-measurement histogram for provenance.
    import hashlib

    digest = hashlib.sha256(
        f"{plan.device_arn}:{plan.shots}:{n_qubits}".encode()
    ).digest()
    counts = {}
    remaining = plan.shots
    n_states = min(2**n_qubits, 16)
    for index in range(n_states):
        weight = digest[index % len(digest)] + 1
        take = max(1, (remaining * weight) // (256 * (n_states - index)))
        take = min(take, remaining)
        bitstring = format(index % (2**n_qubits), f"0{n_qubits}b")
        counts[bitstring] = take
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        counts["0" * n_qubits] = counts.get("0" * n_qubits, 0) + remaining
    return {
        "status": "COMPLETED",
        "mode": "local_emulate",
        "task_arn": None,
        "device_arn": plan.device_arn,
        "shots": plan.shots,
        "estimated_cost_usd": estimated,
        "measurement_counts": counts,
        "note": "Emulated locally; not an AWS Braket task. Set BRAKET_SUBMIT=1 with credentials for live SV1.",
    }


def submit_quantum_task(plan: BraketJobPlan, poll: bool = True, timeout_s: float = 300.0) -> dict[str, Any]:
    """Submit a qubit toy circuit to Braket. CV analogue tasks should use Hybrid Jobs."""
    estimated = _assert_live_gates(plan)
    if plan.task == "cv_sensing_analogue":
        raise RuntimeError(
            "cv_sensing_analogue is classical; use hybrid E1–E4 or attach classical shadow"
        )
    if plan.task == "hybrid_e1_e4":
        raise RuntimeError("Use submit_hybrid_job helpers for hybrid_e1_e4")

    from braket.aws import AwsDevice
    from braket.tracking import Tracker

    n_qubits = int(plan.parameters.get("n_qubits", 4))
    j_coupling = float(plan.parameters.get("j_coupling", 1.0))
    circuit = build_ising_circuit(n_qubits=n_qubits, j_coupling=j_coupling)
    device = AwsDevice(plan.device_arn)
    s3_folder = (plan.s3_bucket, plan.s3_prefix.rstrip("/") + "/" + plan.job_name)
    with Tracker() as tracker:
        task = device.run(circuit, s3_destination_folder=s3_folder, shots=plan.shots)
        task_arn = task.id
        result_payload: dict[str, Any] = {
            "status": task.state().value if hasattr(task.state(), "value") else str(task.state()),
            "mode": "live",
            "task_arn": task_arn,
            "device_arn": plan.device_arn,
            "shots": plan.shots,
            "estimated_cost_usd": estimated,
            "s3_folder": list(s3_folder),
        }
        if poll:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                state = task.state()
                state_name = state.value if hasattr(state, "value") else str(state)
                if state_name in {"COMPLETED", "FAILED", "CANCELLED"}:
                    result_payload["status"] = state_name
                    break
                time.sleep(2.0)
            else:
                result_payload["status"] = "TIMEOUT"
            if result_payload["status"] == "COMPLETED":
                result = task.result()
                result_payload["measurement_counts"] = dict(result.measurement_counts)
        try:
            result_payload["tracker_cost_usd"] = float(tracker.qpu_tasks_cost()) + float(
                tracker.simulator_tasks_cost()
            )
        except Exception:  # noqa: BLE001 - tracker optional
            result_payload["tracker_cost_usd"] = None
    return result_payload


def credentials_available() -> bool:
    """Return True if boto3 can resolve any AWS credentials."""
    try:
        import boto3

        session = boto3.Session()
        return session.get_credentials() is not None
    except Exception:  # noqa: BLE001
        return False
