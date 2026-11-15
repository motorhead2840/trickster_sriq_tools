# Final Dissertation validity — SageMaker g4dn (complete)

**Branch / commit:** `cursor/dissertation-braket-validity-fd3c` @ `9ed38e0`  
**Generated (UTC):** `20261112T161509Z`  
**Host:** SageMaker g4dn (JAX CPU fallback; CUDA-Q present)

## Verdict

| Gate | Result |
|---|---|
| `pass_dissertation_validity` | **true** |
| Enabled math | true |
| Safety-tax architecture | true |
| Lemma Z1 | true |
| User-query path | true |
| Softwall honest | true |
| Chapters 1–6 | **all true** |

| Metric | Value |
|---|---|
| In-process median | **0.495 ms** |
| Post-filter model midpoint | **300 ms** |
| Architectural ratio | **~606×** (modeled baseline) |
| User-path decision | `present_answer` (~0.32 ms, conf ~0.88) |

## Softwalls (held)

- Landauer circumvention: `NOT_PROVEN`
- TMEM MSB-XOR O(0): `SOFTWARE_PROXY_NOT_TMEM`
- Blackwell 0.0062 ms: `REFERENCE_ONLY_NOT_REMEASURED`
- MBL quasicrystal: `INSTRUMENT_ONLY_NOT_PROVEN`
- PPLN / optical (Ch. 6): `FUTURE_EMBODIMENT_NOT_PROVEN`
- Any softwall falsely greened: `false`

## Chapter pastebacks

See [`pasteback-chapters.json`](pasteback-chapters.json) and full suite [`pasteback.json`](pasteback.json).

## Counsel note

Enable ENABLED_* rows only. Treat Exascale / absolute-zero-entropy rhetoric as vision. 300 ms is a modeled filter midpoint, not a matched E2E product baseline.
