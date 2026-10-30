# Models

The neural components of the conditional INR. The decoder is trained in the main loop and frozen at
test time, when only the per-eye latent is optimised.

| File | Contents |
|---|---|
| `inr_decoder.py` | `INR_Decoder`: bilinearly samples the per-eye latent grid at a query coordinate, encodes the conditioning variables, concatenates both into the modulation vector, runs the SIREN trunk, and emits FAF intensity + GA logits. `forward(...)` for training, `inference(...)` for dense grid decoding. Also `Modulator`, an optional convolution over the latent grid (identity when `cnn_kernel_size: 0`). |
| `siren.py` | `SineLayer` and `Siren`: the SIREN MLP trunk with per-feature FiLM (a scale and shift, computed from the modulation vector, inside the sine) and the reconstruction / segmentation heads. |
| `omega_scheduler.py` | `get_omega_schedule`. Per-layer SIREN activation frequency (ω) schedule: `constant`, `linear`, or `exponential`. |
| `encodings.py` | Coordinate / condition / time encoders: hash-grid, Fourier, Gaussian, identity, and MLP condition encodings, with `get_encoding` / `get_condition_encoding` factories. |
