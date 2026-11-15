# Yoga–Buddhi projector — source equations and staged complement

**Sources (uploaded 2026-11-12):**
- *The Singularity-Based Realization Interface (SRI)* dissertation (Thiruvananthapuram, 2026-03-25), Ch. 2–4
- *The Ω-Manifold* master white paper (Feb 2026), §§2–5 (Mirror-Validation / optical phase-conjugate; does not name Yoga–Buddhi)

**Repo stage this complements:** A0–A5, Theorem Zero, Lemma Z1 (involution latent-only), `sri.cv.estimator.observable_yoga_buddhi`.  
See [architecture-axioms.md](architecture-axioms.md).

**Honesty.** Dissertation free-energy / Landauer-circumvention / hardware \(\mathcal O(1)\) narratives are **source claims**. Track A refuted software \(\mathcal O(1)\) and left Landauer circumvention `not_validated`. This note extracts the projector algebra and updates it for the deployable path.

---

## 1. Source ontology (dissertation + master WP)

### 1.1 Ω-dit

\[
|\Omega\rangle
=
\int_{-\infty}^{+\infty}
\Bigl(
\sigma(t)\,e^{i\beta(t)}\,[v(t)+\xi(t)]
\Bigr)\,d\omega
\]

| Symbol | Dissertation gloss | Role |
|---|---|---|
| \(\sigma\) | Sat / existence-density | Amplitude of truth |
| \(\beta\) | Subsistence / pattern | Phase / resonance |
| \(v\) | Vidya | Manifest value (truth-state) |
| \(\xi\) | Avidya | Stochastic flux (noise) |

Master WP writes the same integral as Eq. (1) (with \(\upsilon\) for value).

### 1.2 Bipartite map (A1)

\[
\mathcal{H}_{VH}=\mathcal{H}_V\oplus\mathcal{H}_H
\]

- Vertical \(\mathcal{H}_V\): invariant truth (\(|v\rangle\))
- Horizontal \(\mathcal{H}_H\): stochastic flux (\(|\nabla\xi\rangle\))

### 1.3 Thesis (forward-pass state)

\[
|\Psi_{\mathrm{Thesis}}\rangle
=
|v\rangle+|\nabla\xi\rangle
\]

---

## 2. Source Yoga–Buddhi algebra (dissertation Ch. 3)

### 2.1 Symplectic conjugation

For every horizontal perturbation \(+|\nabla\xi\rangle\) there is an exact antithesis (Liouville / phase-space pairing in the source narrative):

\[
\hat{\mathcal{C}}_S
=
\hat{\Pi}_V-\hat{\Pi}_H
\]

\[
|\Psi_{\mathrm{Antithesis}}\rangle
=
\hat{\mathcal{C}}_S|\Psi_{\mathrm{Thesis}}\rangle
=
|v\rangle-|\nabla\xi\rangle
\]

**Latent involution (locked decision).** When \(\hat{\Pi}_V,\hat{\Pi}_H\) are complementary orthogonal projectors on \(\mathcal{H}_{VH}\),

\[
\hat{\mathcal{C}}_S^{2}=\hat{\mathbb{I}}
\quad\text{on the latent chart }(v,\xi).
\]

This is **oracle structure**: it assumes access to the horizontal component so the sign flip is exact. Deployable maps on raw observations \(Y\) are **not** required to satisfy \(\widehat{\mathcal{C}}_S^{\circ 2}=I\) (Lemma Z1 decision).

### 2.2 Yoga–Buddhi projector

\[
\hat{\Upsilon}_{YB}
=
\frac12\bigl(\hat{\mathbb{I}}+\hat{\mathcal{C}}_S\bigr)
\]

Samya (neutral ground) by superposing thesis and antithesis:

\[
|\Psi_{\mathrm{Samya}}\rangle
=
\hat{\Upsilon}_{YB}|\Psi_{\mathrm{Thesis}}\rangle
=
\frac12\Bigl(
\bigl(|v\rangle+|\nabla\xi\rangle\bigr)
+
\bigl(|v\rangle-|\nabla\xi\rangle\bigr)
\Bigr)
=
|v\rangle
\]

**Projector property (latent).** \(\hat{\Upsilon}_{YB}^{2}=\hat{\Upsilon}_{YB}\) and \(\operatorname{ran}(\hat{\Upsilon}_{YB})\subseteq\mathcal{H}_V\) under the bipartition assumptions above.

### 2.3 Source thermodynamic narrative (not empirically locked)

Dissertation Ch. 4 couples the projector to Helmholtz free energy \(\mathcal{F}=\mathcal{U}-T\mathcal{S}\) and claims \(\operatorname{Var}(\nabla\xi)\to 0\Rightarrow\mathcal{S}\to 0\Rightarrow\mathcal{F}\to\mathcal{U}_{\min}\).  
**Staged stance:** keep as *aspirational thermodynamics*. Do not treat Track A/B artifacts as proof of Landauer circumvention or zero free-energy hardware.

### 2.4 Hardware sketch in the dissertation (embodiment layer)

Bitwise sign-flip on isolated noise mantissas (IEEE-754 MSB / XOR `0x80000000`) is presented as \(\mathcal{O}(0)\) generation of the antithesis **when noise is already isolated in TMEM**. That presupposes a solved separation of \(v\) and \(\xi\) — exactly Theorem Zero’s gap. Optical PPLN phase-conjugate mirrors (master WP §5.2) are the analog embodiment of the same cancellation idea.

---

## 3. Complementary channel — Mirror-Validation (master WP)

Master WP Eq. (2), also yellow-paper Vyāpti companion:

\[
\mathcal{M}(C)=C\cdot\operatorname{conj}(C)\;\longrightarrow\;\sigma_{\mathrm{sat}}^{2}
\]

This is an **amplitude / self-consistency** channel (Dharma-Node saturation), not the symplectic sign-flip. In A0–A5 it sits under **A5 (Vyāpti)**: marks that fail inherent geometry are cancelled; marks that match saturate existence-density.

| Channel | Operator | Cancels | Our axiom |
|---|---|---|---|
| Yoga–Buddhi | \(\hat{\Upsilon}_{YB}=\tfrac12(I+\hat{\mathcal{C}}_S)\) | Horizontal flux via antithesis | A1–A2 |
| Mirror-Validation | \(\mathcal{M}(C)=C\overline{C}\) | Non-inherent complex residue | A5 |

---

## 4. Staged complement (what we update for the current repo)

### 4.1 Two projectors, one name

| Mode | Inputs | Map | Error | Code |
|---|---|---|---|---|
| **Oracle Yoga–Buddhi** | known \(v,\xi\) (or known \(\xi\) with thesis \(v+\xi\)) | exact \(\hat{\Upsilon}_{YB}\) | \(0\) | `oracle_paired_projection`, `symplectic_project` |
| **Observable Yoga–Buddhi** | \(Y=(y_A,y_B)\) only | Lemma Z1 rewrite | MMSE | `observable_yoga_buddhi` |

### 4.2 Lemma Z1 bridge (accepted)

Under Gaussian dual-mode model \(\mathcal{M}_{\mathrm{CV}}\) with Wiener weights \(w=(w_A,w_B)\):

\[
\widehat{\mathcal{C}}_S(y_A,y_B)
=
(2w_A-1)y_A+2w_B\,y_B
\]

\[
\hat{\Upsilon}_{\mathrm{obs}}(Y)
=
\frac12\bigl(y_A+\widehat{\mathcal{C}}_S(Y)\bigr)
=
w_A y_A+w_B y_B
=
\hat s_{\mathrm{MMSE}}
\]

**Reading against the dissertation.**  
- Thesis chart on mode A: \(y_A\) plays the role of the measured thesis coordinate.  
- \(\widehat{\mathcal{C}}_S\) is an *observable antithesis functional*, not a latent \(\hat{\Pi}_V-\hat{\Pi}_H\) with known \(\xi\).  
- Exact Samya \(|v\rangle\) is recovered only in the oracle limit (known \(n\), vanishing vacuum noise). Otherwise Samya is **approximate** with error \(\mathrm{MSE}_{\mathrm{MMSE}}\).

### 4.3 Notation lock

Prefer dissertation symbols in theory docs:

- \(\hat{\mathcal{C}}_S\) — conjugation (latent involution)
- \(\hat{\Upsilon}_{YB}\) — Yoga–Buddhi projector
- \(\widehat{\mathcal{C}}_S\), \(\hat{\Upsilon}_{\mathrm{obs}}\) — observable approximations (Lemma Z1)
- \(|\Psi_{\mathrm{Samya}}\rangle\) — post-projection state (\(=|v\rangle\) when exact)

### 4.4 Provenance update (2026-11-12)

- **C5 confirmed** on SageMaker `ml.g4dn.xlarge` (JAX `cuda:0`): Colab oracle Samya fixture + Lemma Z1 grid + Sphota NumPy/JAX/tile_ref. Pasteback: [sagemaker-g4dn/pasteback.json](benchmark-results/braket-live/sphota-yoga-buddhi-elaborate/sagemaker-g4dn/pasteback.json).
- **Z1b metric:** empirical \(\mathbb E[\lVert\widehat{\mathcal{C}}_S(Y)+y_A\rVert_2^2]\approx 1.28233\) on the elaborate grid (raw-\(Y\) non-involution witness).
- Still does **not** choose Lemma Z2 order (unveil→Z1 vs select→Z1).
- Still does **not** claim hardware MSB-flip / Landauer / cuTile CUPTI geomean.

---

## 5. Minimal identity checklist (for tests / reviews)

1. **Oracle Samya:** \(\tfrac12[(v+\xi)+(v-\xi)]=v\).  
2. **Latent involution:** \(\hat{\mathcal{C}}_S^{2}=I\) on \((v,\xi)\).  
3. **Observable identity:** \(\hat{\Upsilon}_{\mathrm{obs}}\equiv\hat s\) at rtol \(10^{-10}\) (C5 algebra).  
4. **Raw-\(Y\) non-involution:** \(\widehat{\mathcal{C}}_S(Y)\) need not equal \(-y_A\).  
5. **Mirror channel separate:** \(\mathcal{M}\) is not a substitute for \(\hat{\Upsilon}_{YB}\).

---

## 6. Embodiment hooks (NVIDIA stack)

| Path | Role | Doc |
|---|---|---|
| cuTile Python→Rust agentic | Port Samya / \(\hat\Upsilon_{\mathrm{obs}}\) / Mirror tiles with IR-diff + CUPTI gate | [cuda-tile-rust-embodiment.md](cuda-tile-rust-embodiment.md) |
| TE + JAX dropless MoE | Train SREE Mentor MoE backbone without capacity-drop; Witness still pre-Softmax | [sree-mentor-jax-te-moe.md](sree-mentor-jax-te-moe.md) |

C5 is now GPU-stamped; these paths still do **not** claim hardware MSB-flip \(\mathcal O(0)\) or TileGym/DeepSeek numbers.

## 7. Next hinge

Compose dissertation Yoga–Buddhi with A3/A4 marks (**Lemma Z2**): unveil and/or selection supply better antithesis marks so \(\widehat{\mathcal{C}}_S\) approaches latent \(\hat{\mathcal{C}}_S\). That is the path from approximate Samya (MMSE) toward architecture-grade Samya without oracle \(\xi\). Z1/C5 being confirmed on T4 is the prerequisite, not the destination.
