# Legacy code

These modules are kept so older scripts and the committed measurements can still be read. They are not part of the supported `sri` API, and `import sri` does not load them.

## Paired-state projector

`simulator.py` (`symplectic_project`) and `pqc.py` implement

`0.5 * ((v + xi) + (v - xi))`.

Both the clean vector and the noise are supplied by the caller. The projector does not estimate unknown noise from an observation. It only checks the algebra when the noise is already known. `pqc.py` is that same check on a fixed vector. It is not cryptography.

## χ_v3 damping

`v3_damping` in `simulator.py` follows the printed Circuit B formula with `lambda_max` (`Λ_max`) = 50. The factor `exp(-50)` is about `1.9e-22`, so one step drives the vector to the float floor. That removes the signal along with the noise. A near-zero residual is not evidence that the damping distinguished the two. The GPU script `sri_gpu_demo.py` is the same damping on a JAX run; it stays at the repository root because the README reproduces it from that path.

## Ξ as an LLM hallucination detector

`llm_benchmark.py` scores TruthfulQA multiple-choice items with Ξ and compares that score to model confidence. `llm_xi_analysis.py` recomputes the summary. On the committed records the Ξ AUROC is 0.47–0.54, which is chance. Ξ is not a hallucination detector.

The old import paths (`sri_simulator`, `sri_pqc`, `sri_core_engine`, `experiments.llm_benchmark`, `experiments.analyze_llm_results`) are thin shims onto this folder.
