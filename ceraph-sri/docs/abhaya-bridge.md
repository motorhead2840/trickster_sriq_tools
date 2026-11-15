# Abhaya bridge

The bridge note defines two maps this package now runs on CPU, and two optional launches.

| Map | Implementation | Where |
|---|---|---|
| \(\lvert\Omega\rangle \approx \sum \sigma\,\beta\,(\upsilon+\xi)\,\Delta\omega\) | `omega_dit` | `sri/omlang/abhaya.py` |
| \(A(\Theta)=\Theta\exp(-\lambda\Xi)\) | `abhaya_damping` | same |
| \(M(C)=C\cdot\mathrm{conj}(C)\) | `mirror_saturation` | same, via `mirror_validation` |
| Triton tile of \(A\) | `abhaya_triton.launch` | one CUDA launch when Triton and CUDA import |
| JAX primitive of \(A\) | `abhaya_jax.apply` | when JAX imports; JVP is the analytic derivative |
| `@sphota` block ladder | `sphota_schedule` and `abhaya_bridge.jl` | 128, 256, 512, or 1024 from the mean absolute \(\sigma\) |

The Triton kernel loads a tile of \(\Theta\) and \(\Xi\), applies the exponential, and stores \(A\). That is one launch rather than a separate kernel for the exponential. The tile is staged in SRAM. The reads and the write still use device memory. No run in this repository has measured that kernel on an A100 or on the RTX PRO 6000.

The block ladder is a schedule. It is not a measured occupancy. The same edges are in the Julia macro `@sphota`. Julia is invoked as a subprocess with Float32 files, which is the path the rest of `sri.omlang` already uses. PythonCall is not required for that path.

The GCP project, the two A100 instances, and the VPC named in the bridge note are not contacted by this code. A negative \(\Xi\) increases \(A\), because that is what the written exponential does.

Phase 4 still scores device claims from measured rows only. This bridge adds the equation and the kernel. It does not add a Blackwell or A100 timing row.
