# `sri-cutile-rs-stub`

CPU-only mirror of Yoga–Buddhi **tile** ops for the NVIDIA cuTile Python→Rust
agentic path. See [`docs/cuda-tile-rust-embodiment.md`](../../docs/cuda-tile-rust-embodiment.md).

```bash
cd embodiment/cutile_rs_stub
cargo test
cargo run --bin sri_cutile_rs_stub_check
```

Expect `VERDICT: KERNEL_OK`. Real cuTile + Blackwell + CUPTI geomean ≥ 0.95
is a later stage (`PERF_OK`); this stub only clears the functional kernel gate.
