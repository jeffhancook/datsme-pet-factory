# pet_factory

**Type an animal name, get a ready-to-use DatsMe pet.**

```python
from pet_factory import make_pet_zip

breed_id, zip_bytes = make_pet_zip("red panda")
open(f"{breed_id}.zip", "wb").write(zip_bytes)
```

The `.zip` is a complete **DatsMe breed bundle** (a transparent sprite sheet +
`manifest.json` + `package.json`) with an **animal-appropriate set of
animations** — exactly the shape DatsMe's `POST /api/pets/me/upload` already
accepts. It's generated entirely locally on a GPU (no paid APIs).

By default each animal gets a fitting set (an **idle** rest loop plus one or more
motion loops): a bird gets **idle + fly + hop**, a fish **idle + swim**, rabbits
and frogs **idle + hop + walk**, and most mammals **idle + walk**. You can also
pass an explicit list, e.g. `make_pet_zip("owl", animations=["idle","fly"])`.
Available animations: `idle, walk, run, fly, hop, swim`.

Pipeline: `animal → Z-Image base sprite (side profile, facing right) → one Wan
2.2 I2V loop per animation → birefnet background removal → packed .zip`.
Takes **~3–5 minutes** on an RTX 3090 (depending on how many animations).

---

## Why this exists / how to use it in DatsMe

DatsMe's API server has **no GPU**, so it can't run this pipeline directly. Two
ways to integrate:

- **If your backend has a CUDA GPU:** just `pip install` this and call
  `make_pet_zip()` from a background task.
- **If it doesn't (DatsMe's case):** use the **queue + worker** pattern in
  [`examples/`](examples/). Your backend enqueues `{animal}`; a small worker on
  a GPU box polls the queue, runs `make_pet_zip()`, and uploads the `.zip` back.

> **Full isolation-first integration plan:** see
> **[DATSME_INTEGRATION.md](DATSME_INTEGRATION.md)** — concrete file layout,
> route stubs, the feature flag, and a failure-mode table showing DatsMe keeps
> running normally if this feature breaks, is off, or the GPU box is down.

### Wiring the result into DatsMe

The generated `.zip` passes DatsMe's `validate_uploaded_bundle()` unchanged. To
give a user a pet, either:

1. **Reuse the existing upload endpoint** — POST the bytes as multipart field
   `file` to `POST /api/pets/me/upload` (the same endpoint the "⬆ Upload a pet
   bundle (.zip)" button uses). This adopts the pet into the user's house.
   *Heads-up:* adoption may cost credits if `credit_pet_adoption_cost > 0`.
2. **Or write it server-side** — call `pet_assets_service.write_assets(...)`
   with `sheet_png`, `manifest_json`, `package_json` from the zip, skipping the
   HTTP round-trip and the credit charge if you want "make your own pet" to be
   free.

A natural DatsMe feature: a **"Make your own pet"** button in Settings → Pet →
type an animal → (queue job) → poll → the new pet appears in the house. All the
plumbing (submit/status/result + worker) is in `examples/`.

---

## Requirements

The GPU box needs **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**
running with these models and custom nodes:

**Models** (drop into ComfyUI's `models/` folders):
| role | file | folder |
|------|------|--------|
| Z-Image UNet | `zImageTurbo_turbo.safetensors` | `diffusion_models/` |
| Z-Image VAE | `zimage_ae.safetensors` | `vae/` |
| Z-Image text encoder | `qwen_3_4b_fp8.safetensors` | `text_encoders/` |
| Wan 2.2 I2V high-noise | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` |
| Wan 2.2 I2V low-noise | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` |
| Wan 2.1 VAE (for 14B) | `wan_2.1_vae.safetensors` | `vae/` |
| Wan text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` |
| LightX2V 4-step LoRA (high) | `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | `loras/` |
| LightX2V 4-step LoRA (low) | `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | `loras/` |

(Filenames are configurable — see the constants at the top of
`pet_factory/factory.py`.)

**Custom nodes:** [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
(provides `VHS_LoadImagePath`, `SaveAnimatedWEBP` is core).

**System:** `ffmpeg`, ~24 GB VRAM (RTX 3090/4090), Python 3.10+.

## Install

```bash
pip install -e ".[gpu]"      # or ".[cpu]" for CPU-only background removal
sudo apt install ffmpeg
```

Then, with ComfyUI running:

```bash
python examples/cli.py "penguin" -o penguin.zip
```

## GPU cutout (recommended)

Background removal (birefnet) is ~12× faster on the GPU. `onnxruntime-gpu` needs
CUDA 12 + cuDNN 9 libraries at runtime. The easiest source is the ones PyTorch
already bundles — point `LD_LIBRARY_PATH` at them before launching, e.g.:

```bash
NV=/path/to/comfyui/venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:$LD_LIBRARY_PATH"
python examples/worker.py
```

It falls back to CPU automatically if these aren't found (just slower — no crash).

## Configuration (env vars)

| var | default | meaning |
|-----|---------|---------|
| `PET_FACTORY_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `PET_FACTORY_COMFY_OUTPUT` | `~/ComfyUI/output` | ComfyUI's output dir (must be readable by this process) |

## Public API

```python
make_pet_zip(animal, on_progress=None, breed_id=None, animations=None) -> (breed_id, zip_bytes)
pack_datsme_bundle(anims, breed_id, display_name, ...) -> zip_bytes
```

`on_progress(message, fraction)` is called through the run for UI progress.
`animations` is an optional list of preset names (`idle, walk, run, fly, hop,
swim`); omit it to get an animal-appropriate default set. `idle` is always
included and at least one motion animation is guaranteed.

`pack_datsme_bundle`'s `anims` is an ordered list of
`{"name": str, "frames": [PIL images], "role": "rest"|"active"}` — one entry per
animation; each is laid out on its own grid rows.

> **DatsMe runtime note:** the current DatsMe quadruped runtime only carries a
> pet *across the screen* for an animation **named `walk` or `run`**; `fly`,
> `hop`, and `swim` play **in place**. So a bird will flap and hop where it
> stands until DatsMe adds a flying/hopping locomotion strategy. Everything is
> still a valid bundle and plays — this only affects horizontal travel.

## Output format (DatsMe breed bundle)

The `.zip` contains `<breed_id>_sprite.png`, `manifest.json`, `package.json`.
Frames are laid out in a grid; the runtime maps a frame index to a cell as
`col = index % columns`, `row = index // columns`. `manifest.json` looks like:

```json
{
  "schema_version": "pet_manifest.v1",
  "columns": 8, "rows": 6, "frame_width": 256, "frame_height": 256,
  "animations": {
    "idle": { "frames": [0,...,15],  "fps": 12, "loop": true, "runtime_role": "rest",   "rest_dwell_ms": [2000, 5000] },
    "fly":  { "frames": [16,...,31], "fps": 12, "loop": true, "runtime_role": "active", "pick_weight": 1.0 },
    "hop":  { "frames": [32,...,47], "fps": 12, "loop": true, "runtime_role": "active", "pick_weight": 1.0 }
  },
  "view_kind": "side", "native_facing": "right",
  "mirroring_policy": "flip", "movement_class": "mammalian_quadruped"
}
```

Each animation occupies its own grid rows. The `rest` animation is what plays
when the pet is standing still; `active` animations are the motion loops.

## Notes

- **Facing:** the base prompt forces "facing right" because DatsMe authors pets
  facing right and mirrors them for leftward movement.
- **Quality:** background removal uses `birefnet-general-lite`, which keeps even
  white animals (polar bear, swan) solid — plainer models (u2net/isnet) left
  white bodies translucent or hollowed to an outline.
- **Reliability > cleverness:** every frame goes through birefnet. A faster
  flood-fill shortcut was tried and dropped — it had too many hard-to-detect
  failure modes for a hands-off tool.

## License

MIT — see [LICENSE](LICENSE).
