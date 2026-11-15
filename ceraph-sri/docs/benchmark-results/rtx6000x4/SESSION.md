# RTX PRO 6000 × 4 session, 14 November 2026

Four NVIDIA RTX PRO 6000 Blackwell Server Edition boards, 97,887 MiB and 600 W each. PyTorch 2.14.1+cu130. Triton launched on every board. The first JAX import was a CPU jaxlib. After `jax[cuda13]` and a kernel restart, JAX’s backend was `gpu` on `cuda:0` through `cuda:3`, and the same map matched NumPy within 3.8e-6.

## Supported on this machine

- Hybrid stack. Julia `@sphota` chose block 256. Julia, JAX, and Triton each matched the NumPy map within 1e-5. JAX ran on all four GPUs after the CUDA 13 plugin was installed. Triton medians on the four boards were about 0.012 ms at length 16,384.
- Flux Abhaya, latent value withheld. MSE 1.472 versus a moving average at 1.675, at offset 0 and at offset 5. Flux variance remaining 0.117.
- Raw Abhaya at offset 0 only. MSE 0.774. At offset 5 the raw map rises to 18.17.
- Triton sign-bit XOR. Error 0 on all four boards, medians about 0.009–0.011 ms.
- Triton Abhaya from length 8,192 through 1,048,576. Medians stay about 0.009–0.011 ms and match NumPy. Every one is under the 0.2379 ms internal budget. The older Colab medians were 0.0082 ms and 0.0062 ms.
- A one-line PyTorch map is near-flat on all four boards, about 0.017–0.019 ms from 16,384 to 1,048,576 elements. Slopes are about 0.
- Triton block 256 is the fastest of 128, 256, 512, and 1024 (0.0116 ms versus 0.0147, 0.0126, and 0.0129 ms) and is faster than eager PyTorch at 0.0164 ms.
- A fused GPU map at length 4,096 takes 0.021 ms. A Python loop takes 1.040 ms.
- Paired cancellation residual 1.94e-7 when both tensors are inputs.
- Load power about 86 W and 33–36 °C, against idle near 30 W and 28–31 °C.

## Not supported

- Silence gate on window variance. Best F1 is 0.291 at threshold 8.
- Four-way shard of a 4,194,304-element map. One board 0.047 ms, four boards 0.088 ms.
- Flux map versus an average pool at length 1,048,576. Flux 0.064 ms, pool 0.016 ms. Quality and speed are separate: flux won the MSE comparison and lost this timer.
- Raw Abhaya once the carrier mean is 5.

## Recorded, not a win by itself

- Omega-dit sample on the offset-0 series: sigma 1.448, upsilon −0.040, xi 7.056.
- Fibonacci window of 13. Filtered energy was 1.56 times the noise energy.
- V3 circuit printout. Circuit A 1.09, circuit B 50, the saturating cap.
- Dtype medians at one million elements: float32 0.022 ms, float16 0.020 ms, bfloat16 0.020 ms. Float8 storage allocated and was not timed.

## Single board, taken from GPU 0

No second rental is required. GPU 0 ran the same cells as the other three, alone, before any four-board step.

| Measurement on GPU 0 | Value |
|---|---:|
| Idle power / temperature | 30.45 W / 28 °C |
| Triton Abhaya, length 8,192 | 0.0109 ms |
| Triton Abhaya, length 10,000 | 0.0108 ms |
| Triton Abhaya, length 65,536 | 0.0109 ms |
| Triton Abhaya, length 1,048,576 | 0.0107 ms |
| Sign-bit XOR | 0.0104 ms, error 0 |
| PyTorch map, 16,384 through 1,048,576 | 0.0191, 0.0190, 0.0187, 0.0184 ms |
| Log-log slope | −0.0095 |
| One board, 4,194,304 elements | 0.047 ms |
| Load during the four-board loop | 86.6 W, max 33 °C |

The four-board shard at 0.088 ms is a separate row. It does not replace the single-board time of 0.047 ms.

## Not this session

Landauer calorimetry, a lithium-niobate optical core, a material quasicrystal, a hardware witness cluster, and a labeled language exam.

The flat times are launch-bound maps. They do not show that every larger workload stays at 0.01 ms. The 86 W readings are this light loop, not a board driven toward 600 W.
