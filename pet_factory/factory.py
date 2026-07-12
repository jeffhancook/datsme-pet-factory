"""pet_factory.factory — turn an animal name into a ready-to-use DatsMe pet.

    from pet_factory import make_pet_zip
    breed_id, zip_bytes = make_pet_zip("red panda")
    open(f"{breed_id}.zip", "wb").write(zip_bytes)   # -> a DatsMe pet bundle

The .zip is a DatsMe "breed bundle" (sprite sheet + manifest.json + package.json)
— exactly the shape DatsMe's `POST /api/pets/me/upload` accepts. See README.

Pipeline (all local on a CUDA GPU box running ComfyUI):
    animal -> Z-Image base sprite (side profile, facing right, pose per category)
           -> Wan 2.2 I2V loops: idle + walk + run + excited (same base sprite)
           -> birefnet background removal (GPU) -> packed DatsMe .zip

Config via environment variables (all optional):
    PET_FACTORY_COMFY_URL     ComfyUI base URL         (default http://127.0.0.1:8188)
    PET_FACTORY_COMFY_OUTPUT  ComfyUI's output dir     (default ~/ComfyUI/output)
The factory reads generated files from ComfyUI's output dir, so it must run on
the same machine as ComfyUI (shared filesystem).
"""
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageSequence

# ── Config ───────────────────────────────────────────────────────────────────
COMFY_URL = os.environ.get("PET_FACTORY_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_OUTPUT_DIR = Path(os.environ.get(
    "PET_FACTORY_COMFY_OUTPUT", os.path.expanduser("~/ComfyUI/output")))
CLIENT_ID = uuid.uuid4().hex
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

ZIMAGE_UNET = "zImageTurbo_turbo.safetensors"
ZIMAGE_VAE = "zimage_ae.safetensors"
ZIMAGE_TE = "qwen_3_4b_fp8.safetensors"
WAN_UNET_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN_UNET_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
WAN_TE = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
WAN_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# NOTE: no negative prompt anywhere in this pipeline — both Z-Image turbo and
# the Wan lightx2v 4-step loras sample at CFG 1.0, where the negative is
# mathematically ignored. All steering must live in the POSITIVE prompt.

WALK_SUFFIX = (", mouth closed, no facial animation, no chewing, no talking, eyes still, "
               "performing a full walk cycle in place: legs and feet cycling through one "
               "complete stride, body bobbing naturally up and down with each step, classic "
               "looping sprite walk animation, no horizontal movement of the body, no camera "
               "movement, no panning")
IDLE_SUFFIX = (", mouth closed, no facial animation, no chewing, no talking, eyes still, "
               "gentle idle motion: soft breathing, slight sway, a small bob in place, "
               "no walking, no camera movement, no panning")
RUN_SUFFIX = (", mouth closed, no facial animation, eyes still, running in place: legs "
              "cycling quickly through a full run stride, body bobbing energetically with "
              "each stride, fast looping run animation, no horizontal movement of the body, "
              "no camera movement, no panning")
FLY_SUFFIX = (", mouth closed, no facial animation, eyes still, flying in mid-air: wings "
              "beating up and down through full flap cycles, body rising and dipping gently "
              "with each wingbeat, hovering in place, no horizontal movement of the body, "
              "no camera movement, no panning")
HOP_SUFFIX = (", mouth closed, no facial animation, eyes still, hopping in place: crouching "
              "down then springing up and landing in a small bounce, body compressing and "
              "extending, repeating hop cycle, no horizontal movement of the body, "
              "no camera movement, no panning")
SWIM_SUFFIX = (", mouth closed, no facial animation, eyes still, swimming in place: fins and "
               "body gently undulating, tail swishing side to side, floating and gliding "
               "motion, no horizontal movement of the body, no camera movement, no panning")
CRAWL_SUFFIX = (", mouth closed, no facial animation, eyes still, crawling forward in place: "
                "long body and belly undulating in a slithering wave, creeping motion "
                "cycling in place, no horizontal movement of the body, no camera movement, "
                "no panning")
EXCITED_SUFFIX = (", mouth closed, no facial animation, eyes still, excited happy reaction "
                  "in place: quick joyful bouncing and wiggling, energetic but staying in "
                  "one spot, no horizontal movement of the body, no camera movement, "
                  "no panning")

# --- Locomotion model -------------------------------------------------------
# DatsMe's runtime has ONE locomotion strategy (quadruped) and it only carries a
# pet ACROSS THE SCREEN for an animation literally named "walk" or "run"
# (quadruped.ts motionPredicate). So EVERY pet ships exactly idle + walk + run:
# idle is the only still pose, and walk (short/slow) + run (long/fast) both
# actually travel. The manifest names stay "walk"/"run"; what changes per animal
# is the *sprite motion* — a bird hops then flies, a snake slithers, a fish
# swims. That keeps every non-idle animation moving AND valid for DatsMe today.

# Motion primitives: gait key -> (action phrase for the prompt, Wan motion suffix).
GAITS = {
    "walk":       ("walking at a steady pace",                  WALK_SUFFIX),
    "run":        ("running fast",                              RUN_SUFFIX),
    "fly":        ("flying with wings spread, mid-air",         FLY_SUFFIX),
    "hop":        ("hopping in small quick hops",               HOP_SUFFIX),
    "leap":       ("leaping forward in big powerful bounds",    HOP_SUFFIX),
    "swim_slow":  ("swimming slowly, gently gliding",           SWIM_SUFFIX),
    "swim_fast":  ("darting quickly through the water",         SWIM_SUFFIX),
    "crawl_slow": ("slowly crawling forward, body undulating",  CRAWL_SUFFIX),
    "crawl_fast": ("quickly slithering forward, body rippling", CRAWL_SUFFIX),
}

# Per-category idle (rest) action so a resting fish floats rather than "sits".
IDLE_ACTIONS = {
    "default": "sitting calmly, resting",
    "flyer":   "perched and resting, wings folded",
    "swimmer": "hovering in place, gently floating, fins slowly moving",
    "crawler": "coiled and resting, gently swaying",
    "hopper":  "sitting calmly, resting",
}

# Per-category "excited" click-reaction action. DatsMe plays an animation
# literally named "excited" when the user clicks their pet (useClickPetExcited).
EXCITED_ACTIONS = {
    "default": "jumping up and down excitedly, wiggling with joy",
    "flyer":   "fluttering its wings excitedly, bouncing with joy",
    "swimmer": "wiggling excitedly, doing a happy little wobble",
    "crawler": "raising its head and wiggling excitedly",
    "hopper":  "bouncing up and down excitedly",
}

# Per-category BASE SPRITE pose — the single still every loop is animated from.
# "standing" made upright worms out of snakes and standing goldfish; the base
# pose has to match how the animal actually holds itself.
BASE_POSES = {
    "default": "standing",
    "flyer":   "standing perched",
    "swimmer": "in a side-view swimming pose, mid-water",
    "crawler": "low to the ground, long body extended in a gentle curve",
    "hopper":  "sitting upright",
}

# Per-animation playback fps for the DatsMe manifest (the runtime honors
# per-animation fps). Wan generates at 16; playing run at 16 keeps its energy,
# idle slower reads calmer.
ANIM_FPS = {"idle": 10, "walk": 12, "run": 16, "excited": 14}

# How long DatsMe plays the click-triggered excited loop before returning to
# rest (runtime_role "timed" + loop:true -> timed_buffer_ms IS the duration).
EXCITED_PLAY_MS = 2400

# Category -> which primitive drives the walk (short) and run (long) gait. Both
# move the pet across the screen; idle is the only still animation.
GAIT_PROFILES = {
    "default": {"walk": "walk",       "run": "run"},
    "flyer":   {"walk": "hop",        "run": "fly"},
    "swimmer": {"walk": "swim_slow",  "run": "swim_fast"},
    "hopper":  {"walk": "hop",        "run": "leap"},
    "crawler": {"walk": "crawl_slow", "run": "crawl_fast"},
}

# Keyword heuristics -> category. Order matters (checked top to bottom): a "bee"
# hits flyer before crawler, a "seahorse" hits swimmer, etc.
_CATEGORY_KEYWORDS = (
    ("flyer", ("bird", "jay", "robin", "sparrow", "finch", "cardinal", "eagle", "hawk",
               "owl", "falcon", "parrot", "crow", "raven", "dove", "pigeon", "duck",
               "goose", "swan", "seagull", "gull", "hummingbird", "woodpecker", "toucan",
               "flamingo", "peacock", "chickadee", "wren", "bluebird", "magpie", "starling",
               "dragon", "bat", "butterfly", "moth", "bee", "wasp", "dragonfly", "ladybug",
               "fairy", "phoenix", "pegasus", "pterodactyl", "griffin", "gryphon")),
    ("swimmer", ("fish", "shark", "whale", "dolphin", "orca", "octopus", "squid", "jellyfish",
                 "seahorse", "stingray", "ray", "koi", "goldfish", "clownfish", "betta",
                 "guppy", "tuna", "salmon", "crab", "lobster", "shrimp", "seal", "otter",
                 "manatee", "narwhal", "walrus", "turtle", "axolotl")),
    ("hopper", ("rabbit", "bunny", "hare", "frog", "toad", "kangaroo", "wallaby",
                "grasshopper", "cricket", "flea")),
    ("crawler", ("snake", "serpent", "python", "cobra", "viper", "boa", "anaconda", "worm",
                 "earthworm", "snail", "slug", "caterpillar", "lizard", "gecko", "iguana",
                 "chameleon", "salamander", "newt", "eel", "centipede", "millipede", "ant",
                 "beetle", "cockroach", "roach", "scorpion", "spider", "crocodile",
                 "alligator", "komodo")),
)


def _category(animal: str) -> str:
    a = animal.lower()
    for cat, kws in _CATEGORY_KEYWORDS:
        if any(k in a for k in kws):
            return cat
    return "default"


def _animation_plan(animal: str) -> list:
    """Every pet gets idle + walk + run + excited. walk/run are the two gaits
    DatsMe's quadruped runtime actually carries across the screen; their sprite
    motion is tailored to the kind of animal (bird -> hop then fly, snake ->
    slither slow then fast, fish -> swim slow then fast). idle is the rest pose;
    excited is the click-reaction (played when the user clicks their pet).
    Returns an ordered list of {name, action, suffix, role, fps}."""
    cat = _category(animal)
    prof = GAIT_PROFILES[cat]
    plan = [{"name": "idle", "action": IDLE_ACTIONS.get(cat, IDLE_ACTIONS["default"]),
             "suffix": IDLE_SUFFIX, "role": "rest", "fps": ANIM_FPS["idle"]}]
    for gait_name in ("walk", "run"):
        action, suffix = GAITS[prof[gait_name]]
        plan.append({"name": gait_name, "action": action, "suffix": suffix,
                     "role": "active", "fps": ANIM_FPS[gait_name]})
    plan.append({"name": "excited", "action": EXCITED_ACTIONS.get(cat, EXCITED_ACTIONS["default"]),
                 "suffix": EXCITED_SUFFIX, "role": "timed", "fps": ANIM_FPS["excited"]})
    return plan


# rembg session (lazy, reused)
_REMBG = None


def _rembg():
    global _REMBG
    if _REMBG is None:
        from rembg import new_session
        # Prefer the GPU (CUDA) — ~12x faster than CPU. Falls back to CPU if the
        # CUDA libs/GPU aren't available (onnxruntime-gpu ships the CPU provider
        # too), so this never breaks the cutout. CUDA libs are found via
        # LD_LIBRARY_PATH set in run_worker.sh (borrowed from ComfyUI's torch).
        try:
            _REMBG = new_session("birefnet-general-lite",
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            provs = _REMBG.inner_session.get_providers()
            print(f"[pet_factory] rembg providers: {provs}", flush=True)
            return _REMBG
        except Exception as e:
            print(f"[pet_factory] CUDA rembg init failed ({e}); using CPU", flush=True)
        # birefnet-general (SOTA matting) keeps white/light animals SOLID and
        # consistent frame-to-frame. u2net left white bodies translucent or
        # fully transparent on some frames; isnet hollowed them to an outline.
        # Slower (~1-2s/frame on CPU) but the user prioritizes quality.
        _REMBG = new_session("birefnet-general-lite")
    return _REMBG


_REMBG_CPU = None


def _rembg_cpu():
    """Separate CPU-only session, used per-frame when the CUDA session OOMs
    (e.g. ComfyUI hasn't finished releasing VRAM) — CPU birefnet is ~12x
    slower but always correct, which beats failing the job. Kept separate from
    the GPU session so one transient OOM doesn't demote all future frames/jobs
    to CPU."""
    global _REMBG_CPU
    if _REMBG_CPU is None:
        from rembg import new_session
        _REMBG_CPU = new_session("birefnet-general-lite", providers=["CPUExecutionProvider"])
        print("[factory] rembg CPU fallback session created", flush=True)
    return _REMBG_CPU


def _remove_bg(img: Image.Image, session=None) -> Image.Image:
    from rembg import remove
    return remove(img.convert("RGB"), session=session or _rembg())


def _wait_vram_free(target_mb: int = 12000, timeout_s: int = 30):
    """Block until GPU memory drops below target_mb (ComfyUI's /free releases
    asynchronously and can take ~8s). Falls back to a fixed sleep if nvidia-smi
    is unavailable. Prevents the birefnet CUDA session from OOMing against
    still-loaded Wan models."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            used = int(out.stdout.strip().splitlines()[0])
            if used < target_mb:
                return
        except Exception:
            time.sleep(4)          # no nvidia-smi — best-effort fixed wait
            return
        time.sleep(1.0)


# ── ComfyUI workflows ────────────────────────────────────────────────────────

def _static_image_wf(prompt, seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": ZIMAGE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMAGE_TE, "type": "lumina2"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0],
            "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["2", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "petfactory_still"}},
    }


def _loop_wf(prompt, start_image_path, seed, length=17, fps=16, width=704, height=704):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN_UNET_HIGH, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": WAN_UNET_LOW, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": WAN_TE, "type": "wan"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": WAN_VAE}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": WAN_LORA_HIGH, "strength_model": 1.0}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": WAN_LORA_LOW, "strength_model": 1.0}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 8.0}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["6", 0], "shift": 8.0}},
        "9": {"class_type": "VHS_LoadImagePath", "inputs": {"image": start_image_path, "custom_width": 0, "custom_height": 0}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "12": {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive": ["10", 0], "negative": ["11", 0], "vae": ["4", 0],
            "width": width, "height": height, "length": length, "batch_size": 1,
            "start_image": ["9", 0], "end_image": ["9", 0]}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["7", 0], "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["12", 1],
            "latent_image": ["12", 2], "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["8", 0], "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["12", 1],
            "latent_image": ["13", 0], "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "16": {"class_type": "SaveAnimatedWEBP", "inputs": {
            "images": ["15", 0], "filename_prefix": "petfactory_loop", "fps": float(fps),
            "lossless": False, "quality": 90, "method": "default"}},
    }


def _run(wf: dict, timeout: int = 360) -> str:
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf, "client_id": CLIENT_ID}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected workflow: {r.text[:200]}")
    pid = r.json()["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=10).json()
        for o in h.get(pid, {}).get("outputs", {}).values():
            picks = (o.get("gifs") or []) + (o.get("images") or [])
            if picks:
                return picks[0]["filename"]
        time.sleep(1.5)
    raise TimeoutError("ComfyUI generation timed out")


def _wait_stable(path: Path, tries: int = 30):
    """Wait until the file size stops changing — anim_studio may be re-encoding
    (stabilizing) it in place; reading mid-write yields a corrupt image."""
    last = -1
    for _ in range(tries):
        if path.exists():
            sz = path.stat().st_size
            if sz > 0 and sz == last:
                return
            last = sz
        time.sleep(0.4)


def _frames_rgba(path: Path) -> list:
    _wait_stable(path)
    last_err = None
    for _ in range(6):                       # retry around any residual write race
        try:
            if path.suffix.lower() in VIDEO_EXTS:
                tmp = Path(tempfile.mkdtemp(prefix="pff_"))
                try:
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
                                    str(tmp / "f_%05d.png")], check=True)
                    return [Image.open(p).convert("RGBA") for p in sorted(tmp.glob("f_*.png"))]
                finally:
                    import shutil
                    shutil.rmtree(tmp, ignore_errors=True)
            im = Image.open(path)
            return [fr.convert("RGBA") for fr in ImageSequence.Iterator(im)]
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise last_err


def _fill_holes_alpha(alpha: Image.Image, thr: int = 160) -> Image.Image:
    """Given an alpha matte (L), make interior transparent regions (low alpha NOT
    connected to the image border) fully opaque — fixes rembg punching holes /
    leaving semi-transparent patches inside the animal. Real background (reachable
    from the border) stays transparent. Returns a corrected alpha (L).
    Works on the mask only, so the caller composites it over the ORIGINAL RGB —
    filled holes keep the animal's real color instead of turning black."""
    from collections import deque
    a = np.array(alpha.convert("L"))
    h, w = a.shape
    transp = a < thr
    reached = np.zeros((h, w), bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if transp[y, x] and not reached[y, x]:
                reached[y, x] = True; dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if transp[y, x] and not reached[y, x]:
                reached[y, x] = True; dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and transp[ny, nx] and not reached[ny, nx]:
                reached[ny, nx] = True; dq.append((ny, nx))
    holes = transp & ~reached
    if holes.any():
        a = a.copy()
        a[holes] = 255
    return Image.fromarray(a, "L")


def _fit_square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGBA")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    cell = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cell.paste(img, ((size - nw) // 2, (size - nh) // 2), img)
    return cell


def _slug(animal: str) -> str:
    s = "_".join(animal.lower().split())
    return ("".join(c for c in s if c.isalnum() or c in "_-")[:40]) or "pet"


def _base_prompt(animal: str) -> str:
    # Pose per category (a goldfish shouldn't "stand"), and shadows are banned in
    # the POSITIVE prompt — at CFG 1.0 a negative prompt would be ignored, and a
    # painted ground shadow survives the birefnet cutout as a gray smudge.
    pose = BASE_POSES.get(_category(animal), BASE_POSES["default"])
    return (f"a cute cartoon {animal}, side profile view, facing right, {pose}, "
            "soft pastel colors, muted palette, simple flat shading, "
            "floating on a plain pure white background, no shadow, no ground shadow, "
            "no reflection, storybook style")


def _cutout_cells(frames, frame_size):
    """birefnet cutout + fit each frame into a square transparent cell.

    A failed cutout FAILS THE JOB (after one retry) rather than falling back to
    a fully-opaque alpha — that fallback would silently ship the whole white-
    background frame as an ugly opaque square, the exact "shipped broken pet"
    class the flood-key hybrid was reverted over. The worker reports the error
    to the queue, so failing loudly here surfaces as a clean job error."""
    out = []
    for i, fr in enumerate(frames):
        orig = fr.convert("RGB")
        try:
            a = _remove_bg(orig).convert("RGBA").split()[3]   # birefnet matte
        except Exception:
            # Most likely the CUDA session OOMed against ComfyUI's VRAM.
            # Retry on a CPU-only session (slow but always correct); only a
            # CPU failure — something truly broken — fails the job.
            try:
                a = _remove_bg(orig, session=_rembg_cpu()).convert("RGBA").split()[3]
            except Exception as e:
                raise RuntimeError(f"background removal failed on frame {i}: {e}") from e
        result = orig.convert("RGBA")
        result.putalpha(a)
        cell = _fit_square(result, frame_size)
        cell.putalpha(_fill_holes_alpha(cell.split()[3]))     # close interior holes
        out.append(cell)
    return out


def pack_datsme_bundle(anims, breed_id, display_name, frame_size=256, columns=8,
                       fps=12, movement_class="mammalian_quadruped") -> bytes:
    """Pack N animations into one DatsMe breed bundle (.zip bytes).

    `anims` = ordered list of {"name": str, "frames": [PIL images], "role": str,
    "fps": int (optional, defaults to `fps`)}. Each animation's frames are cut
    out, laid on a fresh grid row, and given frame indices; the manifest maps
    col = idx % columns, row = idx // columns. Roles are DatsMe runtime_roles:
    "rest" plays when idle, "active" while moving, "timed" plays for
    timed_buffer_ms then returns to rest (used for the click-reaction).
    """
    placed = []            # (index, cell) across all animations
    manifest_anims = {}
    cursor = 0
    for a in anims:
        cells = _cutout_cells(a["frames"], frame_size)
        start = ((cursor + columns - 1) // columns) * columns   # each anim starts on a new row
        idx = list(range(start, start + len(cells)))
        placed.extend(zip(idx, cells))
        entry = {"frames": idx, "fps": a.get("fps", fps), "loop": True,
                 "runtime_role": a["role"],
                 "view": {"view_kind": "side"}}   # lets quadruped apply travel tilt
        if a["role"] == "rest":
            entry["rest_dwell_ms"] = [2000, 5000]
        elif a["role"] == "timed":
            # loop:true + timed -> timed_buffer_ms IS the play duration before
            # the runtime returns the pet to rest (useAutoStateMachine).
            entry["timed_buffer_ms"] = EXCITED_PLAY_MS
        manifest_anims[a["name"]] = entry
        cursor = start + len(cells)
    rows = (cursor + columns - 1) // columns

    sheet = Image.new("RGBA", (columns * frame_size, rows * frame_size), (0, 0, 0, 0))
    for i, cell in placed:
        sheet.paste(cell, ((i % columns) * frame_size, (i // columns) * frame_size), cell)

    manifest = {
        "schema_version": "pet_manifest.v1",
        "columns": columns, "rows": rows, "frame_width": frame_size, "frame_height": frame_size,
        "animations": manifest_anims,
        "view_kind": "side", "native_facing": "right",
        "mirroring_policy": "flip", "movement_class": movement_class,
    }
    package = {"breed_id": breed_id, "display_name": display_name, "movement_class": movement_class}

    sbuf = io.BytesIO(); sheet.save(sbuf, "PNG")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{breed_id}_sprite.png", sbuf.getvalue())
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("package.json", json.dumps(package, indent=2))
    return buf.getvalue()


def make_pet_zip(animal: str, on_progress=None, breed_id=None):
    """animal -> (breed_id, zip_bytes). Every pet gets idle + walk + run +
    excited, with sprite motion tailored to the kind of animal (bird hops then
    flies, snake slithers, fish swims). walk and run are the two gaits DatsMe's
    quadruped runtime actually carries across the screen; excited is the
    click-reaction DatsMe triggers when the user clicks their pet. All four
    loops come from one shared base sprite."""
    def prog(msg, pct):
        if on_progress:
            on_progress(msg, pct)

    animal = (animal or "").strip()[:60] or "pet"
    seed = random.randint(1, 2**31)
    plan = _animation_plan(animal)               # idle + walk + run, tailored per animal

    prog("Drawing the base sprite…", 0.08)
    base = COMFY_OUTPUT_DIR / _run(_static_image_wf(_base_prompt(animal), seed))
    _wait_stable(base)

    # One Wan loop per animation, all from the same base sprite.
    loop_files = []
    for i, step in enumerate(plan):
        prog(f"Animating the {step['name']}…", 0.10 + 0.72 * (i / len(plan)))
        loop_files.append(_run(_loop_wf(
            f"cute cartoon {animal} {step['action']}, side profile, facing right" + step["suffix"],
            str(base), seed)))

    prog("Cutting out backgrounds & packing…", 0.85)
    # Unload ComfyUI's Wan models so the GPU has room for birefnet.
    try:
        requests.post(f"{COMFY_URL}/free", json={"unload_models": True, "free_memory": True}, timeout=10)
    except Exception:
        pass
    _wait_vram_free()

    anims = []
    for step, fn in zip(plan, loop_files):
        frames = _frames_rgba(COMFY_OUTPUT_DIR / fn)
        if len(frames) > 1:                      # drop the duplicated final loop frame
            frames = frames[:-1]
        anims.append({"name": step["name"], "frames": frames,
                      "role": step["role"], "fps": step["fps"]})

    breed_id = breed_id or _slug(animal)
    zip_bytes = pack_datsme_bundle(anims, breed_id, animal.title())
    prog("Done!", 1.0)
    return breed_id, zip_bytes
