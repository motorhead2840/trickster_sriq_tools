# Patent annex — enablement & softwalls

## Enabled claim sketches (from R7)

[
  {
    "title": "Observable Yoga\u2013Buddhi / Lemma Z1",
    "summary": "Method of constructing an observable antithesis from dual-mode homodyne outcomes (y_A, y_B) and Wiener weights such that \u03a5_obs = \u00bd(y_A + \u0108_S) equals the MMSE estimator without receiving s or n.",
    "evidence_rounds": [
      "R1",
      "R2",
      "R3",
      "R4"
    ],
    "status": "supported_by_gpu_c5"
  },
  {
    "title": "Sphota multi-backend crystallization",
    "summary": "System and method for compiling Yoga\u2013Buddhi algebra across NumPy/JAX (and optional Julia) with tile-shaped reference ops for later cuTile port.",
    "evidence_rounds": [
      "R1",
      "prior_c5_pasteback"
    ],
    "status": "supported_scaffold"
  },
  {
    "title": "Pre-Softmax Witness gate (software)",
    "summary": "Method of measuring logit stability (\u039e_SNR) and applying damping before tokenization \u2014 SREE Mentor intercept at host/software layer.",
    "evidence_rounds": [
      "R6"
    ],
    "status": "supported_smoke_not_tmem"
  }
]

## Softwall claim board

{
  "software_O1_or_O0": {
    "claim": "Software-native Yoga\u2013Buddhi / projector is O(1) or O(0) in N",
    "instrument": "log-log wall-time vs N (omega_landauer complexity suite)",
    "projector_slope": 0.947179796626127,
    "verdict": "REFUTED_FOR_THIS_SOFTWARE",
    "note": "Software path scales with N (O(N)-like). This refutes treating the present CPU/Python implementation as hardware-native O(1)."
  },
  "tmem_msb_xor_O0": {
    "claim": "IEEE-754 MSB XOR in TMEM generates antithesis at O(0)",
    "instrument": "code search + runtime path (observable_yoga_buddhi uses arithmetic)",
    "verdict": "NOT_INSTRUMENTED_NO_TMEM_XOR_PATH",
    "note": "No CUDA TMEM XOR kernel in this repo path. Softwall stands until a dedicated embodiment implements and measures it."
  },
  "landauer_circumvention": {
    "claim": "Architecture circumvents Landauer / reaches F=U_min as measured fact",
    "instrument": "landauer_minimum_heat theory table + no on-die calorimetry",
    "verdict": "NOT_PROVEN",
    "landauer_rows": [
      {
        "temperature_kelvin": 1.0,
        "kT_ln2_joules": 9.569929616929079e-24,
        "hypothetical_erasures": [
          {
            "bits": 1,
            "ideal_lower_bound_joules": 9.569929616929079e-24,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 1024,
            "ideal_lower_bound_joules": 9.799607927735377e-21,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 16384,
            "ideal_lower_bound_joules": 1.5679372684376603e-19,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          }
        ]
      },
      {
        "temperature_kelvin": 77.0,
        "kT_ln2_joules": 7.368845805035391e-22,
        "hypothetical_erasures": [
          {
            "bits": 1,
            "ideal_lower_bound_joules": 7.368845805035391e-22,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 1024,
            "ideal_lower_bound_joules": 7.54569810435624e-19,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 16384,
            "ideal_lower_bound_joules": 1.2073116966969984e-17,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          }
        ]
      },
      {
        "temperature_kelvin": 300.0,
        "kT_ln2_joules": 2.870978885078724e-21,
        "hypothetical_erasures": [
          {
            "bits": 1,
            "ideal_lower_bound_joules": 2.870978885078724e-21,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 1024,
            "ideal_lower_bound_joules": 2.9398823783206133e-18,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 16384,
            "ideal_lower_bound_joules": 4.703811805312981e-17,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          }
        ]
      },
      {
        "temperature_kelvin": 350.0,
        "kT_ln2_joules": 3.349475365925178e-21,
        "hypothetical_erasures": [
          {
            "bits": 1,
            "ideal_lower_bound_joules": 3.349475365925178e-21,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 1024,
            "ideal_lower_bound_joules": 3.4298627747073822e-18,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          },
          {
            "bits": 16384,
            "ideal_lower_bound_joules": 5.4877804395318116e-17,
            "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat."
          }
        ]
      }
    ],
    "note": "Suite tabulates k_B T ln 2 as a lower-bound theory check only. No joule meter on the GPU die in this campaign."
  },
  "deepseek_tflops_as_ours": {
    "claim": "Ceraph/SREE training stack achieves NVIDIA DeepSeek-V3 TE TFLOPS",
    "instrument": "literature reference table only",
    "verdict": "REFERENCE_ONLY_NOT_OURS",
    "reference": {
      "source": "https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/",
      "model": "DeepSeek-V3 671B (NVIDIA MaxText + Transformer Engine)",
      "baseline_tflops_per_gpu": 103,
      "optimized_tflops_per_gpu": 1068,
      "speedup": 10.4,
      "scaling_efficiency_at_1024_gpus": 0.97,
      "ceraph_status": "REFERENCE_ONLY_NOT_OURS"
    },
    "note": "Numbers are NVIDIA's published MaxText+TE results. Do not paste as Ceraph measurements. TE MoE is a planned Mentor training path."
  },
  "gpu_backend_present": false,
  "any_softwall_falsely_greened": false
}

## DeepSeek reference (NVIDIA; not Ceraph)

{
  "source": "https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/",
  "model": "DeepSeek-V3 671B (NVIDIA MaxText + Transformer Engine)",
  "baseline_tflops_per_gpu": 103,
  "optimized_tflops_per_gpu": 1068,
  "speedup": 10.4,
  "scaling_efficiency_at_1024_gpus": 0.97,
  "ceraph_status": "REFERENCE_ONLY_NOT_OURS"
}

## Figure index

{
  "hist3d_z1_mse_cube": "docs/benchmark-results/braket-live/grand-theory-audit/figures/hist3d_z1_mse_cube.png",
  "hist3d_residual_field": "docs/benchmark-results/braket-live/grand-theory-audit/figures/hist3d_residual_field.png",
  "hist3d_n_scaling_softwall": "docs/benchmark-results/braket-live/grand-theory-audit/figures/hist3d_n_scaling_softwall.png"
}

