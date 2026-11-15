# Spec-session measurement

Private kit for one rented NVIDIA GPU. It records device identity, optional board power and temperature, CPU scaling, a synthetic denoising check, and the existing GPU demo when a supported GPU is visible.

It does not produce a data-center power bill by itself. Keep the output directory private.

## Budget

Use a single GPU for about 3 hours, then stop. At the November 2026 marketplace range for an RTX PRO 6000 Blackwell, that is roughly $6–$10. A second, different GPU for about 2 hours the next day is optional and is the only reason to approach $50. Do not launch an 8-GPU DGX node for this session. A $50–$100 credit is enough for the single-GPU protocol with a reserve, and it is not enough for a full DGX cluster day.

## Run

```bash
python experiments/spec_session/measure.py --output-dir spec_session_out
```

Before trusting a power number, confirm the probe prints a numeric `power.draw`. If it is blank, stop spending. The useful files are `spec_session.json` and `nvidia_smi.csv`.
