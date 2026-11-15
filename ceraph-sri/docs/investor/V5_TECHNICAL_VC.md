# Technical note for investors

SRI version 5. Ceraph Corporation, Kerala, India. 15 November 2026. chief@ceraphco.com. www.ceraphco.com.

Planning figures are not revenue and not an offer of securities. Version 5 is the current design. It withdraws earlier wording that described the same work as a constant-time quantum cancellation, a bypass of a thermodynamic limit, or a finished filter of false sentences.

## Abstract

Version 5 is a classical calibration-and-control architecture. The unit is a tile that estimates its own error, corrects by a gain, and checks itself against a neighbor or a reference. The cancellation identity, an average of a value and a sign-flipped residual, is exact only when the residual is already known. On that oracle case, a November measurement on an NVIDIA RTX PRO 6000 produced variance ratios of 26, 101, 627, 2,500, and 10,026 at noise settings 5, 10, 25, 50, and 100. Those ratios are 1 plus the square of the noise setting when the structure has unit variance. The time on that batch stayed near 0.022 milliseconds. A second map keeps a local level and shrinks only the wiggle. Its error did not change when a temperature record was shifted by 5 °C. A goal that the wiggle fall by half was missed.

Thirty-two mathematical tests of the design, run on a CPU in November, passed 24 in full and failed one pre-registered check in each of the other eight. Every passing test is known mathematics. No quantum speedup was found. The combination of these parts has not yet been compared with a standard calibration stack on hardware. That comparison is the next experiment.

The company is described in the November venture papers as bootstrapped at about 5,000 dollars. Those papers state a seed of 7.5 to 10 million dollars and a planning revenue path from 1.2–1.8 million in year 1 to 9–12 million in year 3. An earlier memorandum the same month used a working ask of 1.25 million dollars, sized to a library and one partner. Version 5 can be started on the smaller plan. The larger plan is the stack, two product intentions, and a later chip study. Neither figure is booked.

## What the design is

Each tile reads a mixture, estimates a residual or a relative veil, and subtracts a gain times the estimate. Neighbors compare readings. A common gain and a common phase are invisible to those comparisons, so the absolute veil is not observable. One known-good reference pins it. The reference must be independent of the data channel and of the actuation path, and it must be refreshed when the veil drifts. A tracker estimates how fast the veil is changing from its own recent misses. An older rule that turned the gain down when neighbors disagreed failed exactly when the veil sped up. The replacement raises the gain when the misses exceed the model.

A tile holds a selected state by a drive against a continuing disturbance. For a simple Markov model the held fidelity is (r + 1/d) / (r + 1), where r is the ratio of the selecting rate to the disturbance rate and d is the number of levels. The rate ratio needed for a fidelity floor of 0.90 is 4.00, 5.67, 6.50, and 7.75 at 2, 3, 4, and 8 levels. Selecting one of eight is slightly harder than selecting one of two. It is not eight times harder. Fidelity above 0.99 was not reached on the scanned rates. The cost in that notebook is a rate, not a joule.

The loop stops by annealing the gain after the misses settle, and it rearms if the target jumps. It hands over by keeping the old state until the new one has locked. It reports error per tile. A training loss that looks finished can hide parameters that are still moving; in one ill-conditioned case the loss settled at step 60 and the parameters at step 28,200.

## What was measured

Oracle table, RTX PRO 6000, float16, batch 16 by 65,536, sixty iterations after warmup:

| Noise setting | Printed ratio | 1 + (noise setting)² | Median time | Cosine to the true structure |
|---|---|---|---|---|
| 5 | 26 | 26 | 0.0223 ms | 1.0000 |
| 10 | 101 | 101 | 0.0218 ms | 1.0000 |
| 25 | 627 | 626 | 0.0218 ms | 0.99999 |
| 50 | 2,500 | 2,501 | 0.0216 ms | 0.99995 |
| 100 | 10,026 | 10,001 | 0.0217 ms | 0.99979 |

The cosine uses the true structure. It is the signature of an oracle test.

Level-preserving map, 14 November, λ = 1. Synthetic error 1.472 across a shift of 5. Melbourne daily-minimum error 5.819 at both a shift of zero and a shift of 5 °C. Raw scaling of the whole reading moved from about 136 to about 267. Variance remaining on that thermometer: 0.515, 0.518, 0.510, against a bar of 0.50. One board, five minutes: 137.3 jobs a second, mean draw 314.5 watts, 32 to 44 °C. Four boards: 549.2 jobs a second, 47 to 48 °C. Long traversal: 1,459.1 GB/s, 91.4 percent of a stated peak of 1,597. A separate 16-byte message was timed at 54.6 GB/s. Those are different measurements.

Relative estimation, CPU notebooks. Ungauged phase error 0.82, gauge-fixed 0.017, on a 64-tile grid. One good reference cut absolute error from a free range of about 0.09–3 to about 0.001. A stale reference let it grow back to 0.49. After a tenfold jump in how fast the veil wanders, a Mehra tracker stayed near the oracle (0.054) and the old gain rule rose to 0.59.

Selection model, notebook v5, CPU, seed 11, about 11 seconds. The rate formula matched. A memory register helped a slow disturbance and stopped helping as the memory forgot faster. A tracker that saw only a noisy fidelity recovered after a jump and chattered when the veil was steady, because its target was exactly the floor.

Thirty-two tests of the surrounding mathematics: 24 full passes. The eight misses are recorded in the Gemini briefing. None of the thirty-two is a new theorem.

## What version 5 does not claim

It does not claim a quantum speedup, a new complexity class, or a hybrid computer between classical and quantum. The tests that touch quantum hardware are textbook: a decoherence-free subspace, a logic lattice, a spectator-qubit feedforward already in the literature, and a Grover search whose advantage died at a few percent dephasing per qubit.

It does not claim that a stability coefficient detects a false sentence. An internal note put that detector at chance, AUROC 0.47 to 0.54. On public sets of 100 items, one later rule left TruthfulQA unchanged at 23 and lowered ARC-Challenge from 50 to 44.

It does not claim a measured saving of Landauer heat. It does not claim latency independent of buffer length. It does not claim that quasiperiodic wiring improved the estimate. It does not claim that one quarter of a tile is the derived protected fraction. A two-qubit protected subspace in the test covered 3/8, and only when most of the noise was shared.

## Costs and revenue

Two plans are on the November record. Version 5 names both, because they buy different things.

The smaller plan, from the 14 November memorandum, is 1.25 million dollars for 18 months, inside a band of 1 to 2 million, with a lean alternative of 700,000 dollars for 12 months. It buys the library, the tests, counsel for a provisional on the claims the attorney brief actually supports, cloud time on the class of board already measured, and the cost of sitting with one design partner. A partner price on the order of 60,000 dollars a year was used for planning and is not signed.

The larger plan, from the November venture report and the technical specification, is 7.5 to 10 million dollars over 18 to 24 months. The stated split is 40 percent engineering, 25 percent intellectual property, 20 percent prototypes, and 15 percent design partners and go-to-market. At 7.5 million that is 3.00, 1.875, 1.50, and 1.125 million. The stated staff is 34 to 62 core technical people plus 15 to 20 support people for each of two products. That headcount is not required to keep the v5 library honest. It is the specification’s number for a software stack, two products, and a chip study. No tape-out quote is in the files. Analog chips cited as precedents were built by other companies.

Stated planning revenue in that larger case: 1.2 to 1.8 million in year 1, with a loss; 4.5 to 6.5 million in year 2; 9 to 12 million in year 3, with operating profit stated at 1.5 to 3.0 million. Year 1 does not carry a 7.5 million dollar raise. The raise is the bridge. No contract is reported as signed. The company is described as holding about 5,000 dollars.

The two product names, an education platform and a device safety suite, are intentions attached to the larger plan. A product that must refuse a false sentence is ahead of the sheets. A product that embeds the estimator, the reference rule, and the level-preserving map in a pipeline a partner already runs is the nearer object, and it is untested on that partner’s data.

## Risks

The oracle ratios will be misread as a property of customer data. They are 1 + σ² for a known residual. The half-variance bar on a real temperature series was missed. The thirty-two tests have not been repeated as a bake-off against Kalman, PID, and periodic recalibration. Several of the eight failed checks were mistakes in the tests, which means the harness needs a new pre-registration before it is cited as a product metric. A hub is cheaper and is a single point of failure. A reference that is not refreshed, or that is itself faulty, ruins the absolute estimate. Deep-tech schedules slip. The philosophical source of the design rules is not evidence that the rules are new, and the attorney brief treats them as known mathematics arranged as a tile discipline.

## Conclusion

Version 5 is a tile that estimates, corrects, and checks its gauge, with an exact cancellation only when the residual is known and a level-preserving shrinkage when the residual is a wiggle around a center. The numbers that support it are the oracle table, the shift test, the gauge test, the selection-rate formula, and a scoreboard that includes its misses. The numbers that do not support it are a sentence filter, a thermodynamic saving, and a quantum speedup. The smaller plan can start the library and one partner. The larger plan is a different company, and it should not be raised on the oracle ratios alone.
