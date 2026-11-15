"""Machine-readable conjecture registry for the CV foundation sprint.

Statuses are restricted to open / confirmed / refuted. Pass criteria must be
written before A100 promotion of a campaign.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Iterable

VALID_STATUSES = frozenset({"open", "confirmed", "refuted"})


@dataclass
class Conjecture:
    conjecture_id: str
    statement: str
    pass_criterion: str
    status: str = "open"
    artifact_paths: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")


def _initial_registry() -> dict[str, Conjecture]:
    return {
        "C1": Conjecture(
            conjecture_id="C1",
            statement=(
                "Yoga–Buddhi with known ξ equals the MMSE estimator only when the "
                "antithesis is physically available; the observable-only Gaussian "
                "MMSE from (y_A, y_B) is the correct physical replacement."
            ),
            pass_criterion=(
                "Oracle paired projection MSE is ~0 when ξ is supplied; observable-only "
                "MMSE matches analytic Wiener MSE within 5σ sampling error and never "
                "receives s or n at inference."
            ),
            status="open",
        ),
        "C2": Conjecture(
            conjecture_id="C2",
            statement=(
                "Under the protocol Gaussian model, squeezed-reference MSE ordering "
                "matches the closed form across independent seeds."
            ),
            pass_criterion=(
                "For default (Vs=Vn=k=1, η=0.8, r=0.7), sample ordering agrees with "
                "analytic ordering on ≥3 seeds; |analytic-sample| ≤ 5σ SE."
            ),
            status="open",
        ),
        "C3": Conjecture(
            conjecture_id="C3",
            statement=(
                "An axiomatic stability coefficient Ξ_SNR (or purity/Fisher proxy) "
                "remains in (0, 1] on all foundation sweep points; legacy Equation-11 "
                "Ξ need not."
            ),
            pass_criterion=(
                "Over the Week-2 spike grid, min(Ξ_SNR) > 0 and max(Ξ_SNR) ≤ 1; "
                "legacy Ξ is reported without clipping."
            ),
            status="open",
        ),
        "C4": Conjecture(
            conjecture_id="C4",
            statement=(
                "Circuit-B Branch W (worked-example thresholding) yields a macro "
                "marker that does not collapse during the noise spike the way legacy "
                "Ξ does; damping corresponds to a CPTP attenuator on covariances."
            ),
            pass_criterion=(
                "Branch W keeps Ξ_SNR in (0,1] through spikes; attenuator_channel "
                "preserves physical covariances; if neither branch satisfies a "
                "stated contraction inequality, status becomes refuted/open with "
                "counterexamples."
            ),
            status="open",
        ),
        "C5": Conjecture(
            conjecture_id="C5",
            statement=(
                "Lemma Z1 (Observable Antithesis, P1): under the Gaussian dual-mode "
                "model with known parameters, there exists an observable map "
                "Ĉ_S(y_A,y_B)=(2w_A-1)y_A+2w_B y_B such that "
                "Υ̂=½(y_A+Ĉ_S) equals the MMSE estimator ŝ. Involution Ĉ_S²=I is "
                "required only in the latent (v,ξ) chart (oracle path), not on raw Y."
            ),
            pass_criterion=(
                "Algebraic identity Υ̂≡ŝ in code on a fixed seed grid (relative "
                "tolerance 1e-10); sample MSE matches analytic MMSE within the same "
                "5σ rule as C1; estimator never receives s or n. Status stays open "
                "until a runner records the result — unit tests may verify the "
                "algebra without promoting the registry artifact."
            ),
            status="open",
        ),
    }


_REGISTRY: dict[str, Conjecture] = _initial_registry()


def reset_registry() -> None:
    """Restore pristine open statuses (for tests and fresh campaigns)."""
    global _REGISTRY
    _REGISTRY = _initial_registry()


def list_conjectures() -> list[Conjecture]:
    return [deepcopy(item) for item in _REGISTRY.values()]


def get(conjecture_id: str) -> Conjecture:
    try:
        return deepcopy(_REGISTRY[conjecture_id])
    except KeyError as error:
        raise KeyError(f"unknown conjecture_id: {conjecture_id}") from error


def record_result(
    conjecture_id: str,
    status: str,
    artifact_paths: Iterable[str] | None = None,
    notes: str = "",
) -> Conjecture:
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    item = _REGISTRY[conjecture_id]
    item.status = status
    if artifact_paths is not None:
        item.artifact_paths = list(artifact_paths)
    if notes:
        item.notes = notes
    return deepcopy(item)


def registry_as_dicts() -> list[dict]:
    return [asdict(item) for item in list_conjectures()]
