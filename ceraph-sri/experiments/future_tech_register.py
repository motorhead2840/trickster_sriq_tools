"""Dissertation and notebook claims adopted as the Witness research program.

Each row is a future technology. ``tonight`` means a DGX software cell can
touch it. ``later`` means the claim is kept and needs another instrument.
Nothing here is marked proven by this file.
"""
from __future__ import annotations

# id, source, tonight_or_later
REGISTER = (
    ("ch1-iterative-pathology", "Dissertation Ch.1", "later"),
    ("ch1-hallucination-is-free-energy", "Dissertation Ch.1", "tonight"),
    ("ch1-safety-tax", "Dissertation Ch.1", "tonight"),
    ("ch1-in-process-damping", "Dissertation Ch.1", "tonight"),
    ("ch2-omega-dit", "Dissertation Ch.2", "tonight"),
    ("ch2-bipartite-manifold", "Dissertation Ch.2", "tonight"),
    ("ch3-fibonacci-cage", "Dissertation Ch.3", "tonight"),
    ("ch3-mbl-material", "Dissertation Ch.3", "later"),
    ("ch3-one-pulse", "Dissertation Ch.3", "tonight"),
    ("ch4-abhaya-xi", "Dissertation Ch.4", "tonight"),
    ("ch4-v3-circuits", "Dissertation Ch.4", "tonight"),
    ("ch4-vyapti-mirror", "Dissertation Ch.4", "tonight"),
    ("ch5-helmholtz-algebra", "Dissertation Ch.5", "tonight"),
    ("ch5-sign-xor", "Dissertation Ch.5", "tonight"),
    ("ch5-latency-0-0062", "Dissertation Ch.5", "tonight"),
    ("ch5-landauer-calorimetry", "Dissertation Ch.5", "later"),
    ("ch6-ppln", "Dissertation Ch.6", "later"),
    ("ch6-riemann-lebesgue", "Dissertation Ch.6", "later"),
    ("nb-abstain", "Raven_Julia", "tonight"),
    ("nb-logits-middleware", "SREE_FOUNDATION / DEEP / SRI", "tonight"),
    ("nb-omega-telemetry", "SREE_FOUNDATION / deepam", "tonight"),
    ("nb-dtype-ladder", "QcuPRGM", "tonight"),
    ("nb-fused-vs-python", "DEEP", "tonight"),
    ("nb-truth-vault", "SREE_FOUNDATION / DEEP", "later"),
    ("nb-witness-cluster", "QcuPRGM", "later"),
    ("nb-ite-hamiltonian", "PyTorch Test Harness", "tonight"),
    ("nb-sdk", "deepam", "tonight"),
    ("paired-cancellation", "DISSERTATION.ipynb / Patent.ipynb", "tonight"),
    ("flux-abhaya", "bridge note + dissertation cells", "tonight"),
    ("thermal-headroom", "14 Nov boards + Ch.1", "tonight"),
    ("sublinear-scaling", "L40S and Blackwell sessions", "tonight"),
    ("sphota-schedule", "omlang bridge", "tonight"),
)


def by_when(when: str) -> list[str]:
    return [row[0] for row in REGISTER if row[2] == when]


def sources() -> set[str]:
    return {row[1] for row in REGISTER}
