# Technical specification

**Identifier.** `v8_final_provisional_run`, as logged on 15 November 2026.

**Scope.** This specification states the rules that the logged program carried out, the formulas a second implementation has to match, and the pass conditions the sheets already satisfy. It is the current specification. Earlier laboratory sheets are not inputs to these rules.

## 1. Terms

| Term | Meaning in this specification |
|---|---|
| User query | The numeric series presented for correction, plus the request to run the selected map. |
| Center \(\mu\) | Arithmetic mean of the series in the working precision. |
| Spread \(\xi\) | Mean of the absolute deviations from \(\mu\). |
| Map A | \(\mu + (y-\mu)\exp(-\lambda\xi)\), with \(\xi\) held from the input when the map repeats. |
| Map R | \(y\exp(-\lambda\xi)\). Scored, then refused when its shift drift is larger. |
| Drift | Maximum absolute gap between \(f(y+s)\) and \(f(y)+s\). |
| Eight-byte rate | \((8N / t) / 10^{9}\) in GB/s, with \(t\) the median device-event time of the selected copy. |
| Wave | Forty passes of map A on 16,777,216 elements, after five warmup calls. |
| Burn | Repeated copies of map A on 50,000,000 elements until the requested seconds elapse. |
| Fleet ratio | Four-board jobs per second divided by one-board jobs per second, computed from the two finished sheets. |
| Nameplate, rate | 1,597 GB/s, the comparison value stored in the sheet. |
| Nameplate, power | 600 W, the comparison value stored in the sheet. |

The selection field in the JSON is named `center_appreciation`. In this specification that field means the negative of the drift. The map with the algebraically larger score is the map with the smaller drift.

## 2. Precision and seeds

Selection and the CPU closure test use 32-bit floating point. The generator is NumPy's PCG64. Seed 11 is the selection seed and the device parity seed. A second seed is not required by the logged sheets. Every board receives the same seed-11 vector for the parity probe. The sheet must say so, and must set independent physical noise to false.

## 3. Selection procedure

1. Draw or receive \(y\). The logged test uses 8,192 normal values plus 2.
2. Compute \(\mu\) and \(\xi\) in float32.
3. Evaluate map A and map R once on \(y\) and once on \(y+5\), with \(\lambda = 1\).
4. Compute both drifts.
5. Select A when its drift is less than or equal to the drift of R. Otherwise refuse to launch and do not start the device section.
6. Iterate map A forty times with \(\xi\) fixed. Compare with \(\mu + (y-\mu)\exp(-40\xi)\). The CPU gate used by the launch is a maximum absolute residual below \(10^{-5}\). The logged residual was 0.0, which passes.

Logged selection, both machines:

- drift of A: \(9.5367431640625 \times 10^{-7}\)
- drift of R: 2.757575511932373
- forty-pass residual: 0.0
- selected map: A

## 4. Device program

The host uses one CUDA device context per board index from 0 through \(B-1\), where \(B\) is `torch.cuda.device_count()`. The logged values are \(B = 1\) and \(B = 4\).

For each board, copy the seed-11 probe of 262,144 float32 values to the device.

- One pass of map A on the probe, and one pass on the probe plus 5. Store the drift.
- Forty passes with the spread fixed. Store the maximum absolute residual against the closed form.

Launch of the timed kernels uses the PyTorch expression `dst.copy_(mu + (src - mu) * scale)` after `mu`, `xi`, and `scale` have been computed. Those three reductions are outside the timed interval for the eight-byte test. They are inside the forty-pass function, and the spread is still taken once from the buffer that enters that call.

Device parity logged on all five board executions: drift \(2^{-20}\), forty-pass residual 0.0, outputs of the wave finite.

## 5. Eight-byte measurement

Buffer length \(N = 50{,}000{,}000\). Seven timed copies, one warmup inside each timing call. Median of the seven device-event times.

\[
r_{8} = \frac{8N}{t_{\mathrm{ms}}\times 10^{-3}\times 10^{9}}, \qquad r_{16} = 2 r_{8}.
\]

Store \(r_{8}\) as the headline. Store \(r_{16}\) as a synthetic double, with the note that a value of \(r_{16}\) above 1,597 is not a DRAM measurement. Set the Boolean `gbps_16B_is_physical_dram` to whether \(r_{16} \le 1597\). On the logged sheets that Boolean is true because both doubled rates are near 730 GB/s.

Do not replace \(r_{8}\) with a figure from a vendor bandwidth test. Do not log a DRAM-byte claim unless a device counter of bytes is also stored. These sheets do not store that counter.

Logged medians and rates are in the patent draft table and are repeated in the technical report. An implementation matches this specification on these inputs when it reproduces those medians within the same timing method, or when a fresh run stores its own seven samples and recomputes the formula above. The formula, not a hardcoded GB/s, is the rule.

## 6. Wave measurement

Length 16,777,216. Passes 40. Warmup 5, unsynchronized except for a device synchronize after the warmup on that board.

Queued time: four successive forty-pass calls on board 0, one device-event pair around the four calls.

Concurrent time: one thread per board, device event recorded inside the thread, host timer started before the first `Thread.start` and stopped after the last `join`.

Bar, only if \(B \ge 4\):

\[
t_{\text{slowest}} < 0.75 \times t_{\text{queued}} \quad \text{and every output is finite.}
\]

If \(B < 4\), store the bar as null. Do not treat null as a failure.

Uncached bound, for the report, not as a bus-counter claim:

\[
\text{bytes} = 16{,}777{,}216 \times 4 \times 40 \times 2, \qquad \text{GB/s} = \text{bytes} / (t_{\text{slowest}} \times 10^{-3}) / 10^{9}.
\]

Store whether that GB/s exceeds 1,597. Logged result: false on both machines.

Store `do_not_conflate_with_300s_rho` as true.

## 7. Burn, power, and heat

Requested duration 300 s, read from the environment variable `CERAPH_BURN_S` when present. Each board:

1. Allocate 50,000,000 float32 values and fill them with normal draws.
2. Compute \(\mu\), \(\xi\), and the scale once.
3. Allocate the destination once.
4. Until the shared deadline, copy the selected expression and increment a local counter.
5. Synchronize the device, then publish the counter.

A sampler thread records, about once a second, power in watts, temperature in degrees Celsius, SM clock in MHz, the throttle-reason mask, the power limit, and whether power exceeds 600 W. The energy counter is read before the sampler starts its useful window and after the workers join. The sheet stores the difference in millijoules.

Pass conditions actually met by the sheets:

- `samples_over_600W` is 0 on every board.
- Throttle-reason mask is 0 on every stored sample.
- Job counts are positive and the duration is within a fraction of a second of 300 s. Logged durations are 300.22318226900006 s and 300.22373933100016 s.

There is no pass condition that requires the mean power, the peak power, or the temperature to hit a target. The specification records them.

## 8. Fleet ratio

Do not compute the ratio inside a single-machine run. Leave `j24_rho_q4_over_q1` null.

After both files exist:

\[
\rho_{\text{rates}} = \frac{q_{4}}{q_{1}} = \frac{J_{4}/T_{4}}{J_{1}/T_{1}}, \qquad \rho_{\text{jobs}} = \frac{J_{4}}{J_{1}}.
\]

For the logged integers \(J_{1} = 236547\), \(J_{4} = 1101630\) and the logged durations, \(\rho_{\text{rates}} = 4.65712080870646\) and \(\rho_{\text{jobs}} = 4.657129449961318\). The two quotients differ because the measured durations differ. Publish both, and publish the machine identities. Do not label either quotient as a single-chassis scaling factor.

## 9. Output schema

A conforming sheet contains:

- `schema`: `v8_final_provisional_run`
- `date_utc`
- `axiology.selected_map`, `rejected_map`, `drift_A`, `drift_R`, `forty_pass_max_err`, `axiology_passed`
- `environment.board_count_B`, `gpu_model`, `pytorch_version`, `python_version`, `pynvml`, `nameplate_w`, `peak_gbps_comparison_nameplate`
- `correctness_per_board[]` with board id, drift, forty-pass residual, same-seed flag, noise flag
- `bandwidth_cells_j25` with median, seven samples, both rates, both percentages, the sixteen-byte Boolean, and the headline
- `wave_scaling_j26` with queued time, per-board events, slowest, host wall, bar, uncached bound, and the conflation flag
- `burn` with duration, requested duration, total jobs, jobs per second, per-board power, temperature, energy, and the time series
- `j24_rho_q4_over_q1`: null on a machine sheet

The time series may be removed in a summary file. The summary must not be described as containing the one-hertz samples.

## 10. Failure rules

Stop before any device launch if map A is not selected or if the forty-pass CPU residual is not below \(10^{-5}\).

Do not describe the run as a memory-roof result when the headline rate is the eight-byte accounting and no bus-byte counter is present.

Do not describe a null wave bar on a one-board sheet as a failed scaling test.

Do not add the wave ratio to the fleet ratio or substitute one for the other.

## 11. Reference numbers from the logged sheets

These are the values a regression check can pin. A new machine run will differ in wall time, power, and temperature. It should not differ in the seed-11 drift of map A, which is an exact 32-bit residual of \(2^{-20}\) for this test, or in the refusal of map R on this series.

| Check | Value |
|---|---|
| Map A drift, CPU and every logged board | \(9.5367431640625\times 10^{-7}\) |
| Map R drift, CPU | 2.757575511932373 |
| Forty-pass residual, CPU and every logged board | 0.0 |
| One-board eight-byte rate | 369.7240514993388 GB/s at 1.08188796043396 ms |
| Four-board eight-byte rate | 365.3048275705898 GB/s at 1.0949759483337402 ms |
| One-board jobs | 236,547 in 300.22318226900006 s |
| Four-board jobs | 1,101,630 in 300.22373933100016 s |
| Rate ratio | 4.65712080870646 |
| Power samples above 600 W | 0 |
| Throttle-reason mask | 0 on every sample |
