"""
modal_app.py  —  AI Film Pipeline
──────────────────────────────────
Two Modal classes:

  FluxGenerator       FLUX.1-dev  →  reference images + keyframe variants
                      Lightweight CLIP embedding blend for character consistency.
                      GPU: A100-40GB  (comfortable headroom for FLUX bfloat16 + CLIP)

  LTXVideoGenerator   LTX-Video-0.9.5  →  first-frame / last-frame interpolation
                      Uses LTXConditionPipeline + LTXVideoCondition (diffusers ≥ 0.33)
                      GPU: A100-40GB  (201 frames × 768×448 needs ~22 GB VRAM)

Deploy:   modal deploy modal_app.py
Serve:    modal serve  modal_app.py     (hot-reload; keeps containers warm)
Logs:     modal app logs ai-film-pipeline

─────────────────────────────────────────────────────────────────────────────
CHANGELOG vs. previous version
─────────────────────────────────────────────────────────────────────────────
BUG FIXES
  1. ImportError: LTXVideoConditioningItem
     The class does not exist in any diffusers release.
     Fix: replaced with `LTXVideoCondition` from
          `diffusers.pipelines.ltx.pipeline_ltx_condition`
          This was introduced in diffusers 0.33.0 — version pin updated.

  2. Wrong pipeline class for dual-frame conditioning
     `LTXImageToVideoPipeline` only accepts a single first-frame image.
     Fix: replaced with `LTXConditionPipeline` which accepts a `conditions`
          list of `LTXVideoCondition` objects, each pinned to a frame index.

  3. Resolution 768×432 is invalid for LTX-Video
     LTX-Video requires both dimensions divisible by 32.
     432 ÷ 32 = 13.5  ✗  (would produce a silent tensor-shape mismatch)
     Fix: changed to 768×448  (448 ÷ 32 = 14 ✓, ratio ≈ 16:9)

  4. `.to("cuda")` + `enable_model_cpu_offload()` conflict
     These methods are mutually exclusive — the explicit `.to()` call breaks
     the offload hooks that `enable_model_cpu_offload()` installs.
     Fix: removed the rogue `.to("cuda")` call; offload is set up cleanly.

IMPROVEMENTS
  5. Upgraded LTX model to LTX-Video-0.9.5
     Better motion quality, improved prompt following, same pipeline API.

  6. Updated frame_rate to 25 fps (LTX-Video-0.9.5 native)
     num_frames changed from 97 → 201  (201 frames ÷ 25 fps = 8.04 s)
     201 satisfies the LTX constraint: (num_frames − 1) % 8 == 0  ✓

  7. Corrected guidance_scale for LTX-Video-0.9.5
     The 0.9.5 model works best at guidance=3.0, not 7.5.
     High guidance causes over-saturation and motion jitter.

  8. Added decode_timestep + decode_noise_scale
     These LTX-specific parameters sharpen detail during VAE decoding.
     Recommended defaults: decode_timestep=0.05, decode_noise_scale=0.025.

  9. Added enable_vae_tiling() to LTXVideoGenerator
     Reduces peak VRAM during decode by processing tiles instead of full frames.
     Essential at 768×448 with 201 frames.

  10. Upgraded LTX GPU from A10G → A100-40GB
      A10G (24 GB) is too tight for 201 frames at 768×448 (~22 GB required).
      A100-40GB provides a reliable 18 GB headroom, eliminates OOM crashes.

  11. Replaced imageio BytesIO export with a temp-file + export_to_video approach
      imageio's in-memory MP4 writer is brittle across versions. Using diffusers'
      built-in export_to_video with a /tmp path is reliable and well-tested.

  12. Fixed FluxGenerator device conflict (same as fix 4)
      Removed `.to("cuda")` / `enable_model_cpu_offload()` conflict for FLUX too;
      FLUX on A100-40GB can fit fully in VRAM, so `.to("cuda")` is used directly.
─────────────────────────────────────────────────────────────────────────────
"""

import io
import os
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Container image
# NOTE: diffusers MUST be >= 0.33.0 — LTXConditionPipeline and LTXVideoCondition
#       were introduced in that release. Pinning to a known-good minor version
#       avoids silent breakage from future API changes.
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")    # ffmpeg needed by export_to_video
    .pip_install(
        "fastapi[standard]",
        # ── diffusion stack ──────────────────────────────────────────────────
        # MUST be ≥ 0.33.0 for LTXConditionPipeline + LTXVideoCondition
        "diffusers>=0.33.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "torch>=2.4.0",
        "torchvision",
        "torchaudio",
        "safetensors",
        "sentencepiece",
        "Pillow",
        # ── video export ─────────────────────────────────────────────────────
        # imageio is used as a fallback inside export_to_video
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "av",
        # ── IP-Adapter CLIP encoder for character conditioning ────────────────
        "ip-adapter",
    )
)

# Separate volumes so weight I/O doesn't contend
flux_volume = modal.Volume.from_name("flux-weights", create_if_missing=True)
ltx_volume  = modal.Volume.from_name("ltx-weights",  create_if_missing=True)

app = modal.App("ai-film-pipeline", image=image)

WEIGHTS_DIR_FLUX = "/weights/flux"
WEIGHTS_DIR_LTX  = "/weights/ltx"

FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
LTX_MODEL  = "Lightricks/LTX-Video-0.9.5"   # upgraded from base LTX-Video
CLIP_MODEL = "openai/clip-vit-large-patch14"

# ─────────────────────────────────────────────────────────────────────────────
# LTX-Video generation constants
# ─────────────────────────────────────────────────────────────────────────────
# Resolution: both dimensions MUST be divisible by 32.
# 768×448 → 768÷32=24 ✓  448÷32=14 ✓  ratio≈16:9 ✓
LTX_WIDTH      = 768
LTX_HEIGHT     = 448

# 201 frames at 25 fps = 8.04 seconds.
# Constraint: (num_frames − 1) % 8 == 0  →  200 % 8 == 0 ✓
LTX_NUM_FRAMES = 201
LTX_FRAME_RATE = 25        # native fps for LTX-Video-0.9.5

# Guidance: 0.9.5 is sensitive to high values; 3.0 is the recommended default.
LTX_GUIDANCE   = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# FLUX Generator  —  reference images + keyframe variants
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="A100-40GB",
    volumes={WEIGHTS_DIR_FLUX: flux_volume},
    timeout=600,
    allow_concurrent_inputs=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class FluxGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline
        from huggingface_hub import login
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        login(token=os.environ["HF_TOKEN"])

        print("↓ Loading FLUX.1-dev …")
        self.pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=WEIGHTS_DIR_FLUX,
        ).to("cuda")
        # FIX: do NOT call enable_model_cpu_offload() after .to("cuda") —
        # they are mutually exclusive. On A100-40GB, FLUX fits entirely in VRAM.
        print("✓ FLUX.1-dev ready")

        print("↓ Loading CLIP image encoder …")
        self.clip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_MODEL, torch_dtype=torch.float16
        ).to("cuda")
        self.clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
        print("✓ CLIP encoder ready")

    def _encode_ref_image(self, ref_image_bytes: bytes | None):
        """Encode a reference image to a CLIP embedding, or return None."""
        if ref_image_bytes is None:
            return None
        import torch
        from PIL import Image

        img    = Image.open(io.BytesIO(ref_image_bytes)).convert("RGB")
        inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return self.clip_encoder(**inputs).image_embeds  # (1, 768)

    @modal.method()
    def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        width: int = 1024,
        height: int = 576,
        steps: int = 28,
        guidance: float = 3.5,
        character_ref_bytes: bytes | None = None,
    ) -> list[bytes]:
        """
        Generate len(seeds) variants of `prompt`.
        If character_ref_bytes is set, its CLIP embedding is blended (30%) into
        FLUX's pooled text embedding to nudge results toward the reference look.
        Returns a list of raw PNG bytes.
        """
        import torch

        ref_embed = self._encode_ref_image(character_ref_bytes)
        results   = []

        for seed in seeds:
            generator = torch.Generator("cuda").manual_seed(seed)
            extra = {}
            if ref_embed is not None:
                _, pooled, _ = self.pipe.encode_prompt(
                    prompt=prompt, prompt_2=None,
                    device="cuda", num_images_per_prompt=1,
                )
                extra["pooled_prompt_embeds"] = (
                    0.7 * pooled + 0.3 * ref_embed.to(pooled.dtype)
                )

            output = self.pipe(
                prompt=prompt,
                width=width, height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                num_images_per_prompt=1,
                **extra,
            )
            buf = io.BytesIO()
            output.images[0].save(buf, format="PNG")
            results.append(buf.getvalue())

        return results


# ─────────────────────────────────────────────────────────────────────────────
# LTX-Video Generator  —  first-frame / last-frame video interpolation
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    # FIX: upgraded from A10G to A100-40GB.
    # 201 frames at 768×448 bfloat16 requires ~22 GB VRAM.
    # A10G (24 GB) is dangerously close; A100-40GB gives a comfortable buffer.
    gpu="A100-40GB",
    volumes={WEIGHTS_DIR_LTX: ltx_volume},
    timeout=900,            # 15-minute hard cap; generation takes ~3–5 min on A100
    concurrency_limit=10,   # up to 10 parallel containers for 10 simultaneous clips
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        # FIX: LTXConditionPipeline replaces LTXImageToVideoPipeline.
        # Only LTXConditionPipeline supports the `conditions` parameter that
        # accepts LTXVideoCondition objects with arbitrary frame_index values.
        from diffusers import LTXConditionPipeline
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

        print(f"↓ Loading {LTX_MODEL} …")
        self.pipe = LTXConditionPipeline.from_pretrained(
            LTX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=WEIGHTS_DIR_LTX,
        )
        # FIX: do NOT call .to("cuda") before enable_model_cpu_offload().
        # enable_model_cpu_offload() installs its own device hooks; calling
        # .to("cuda") first silently corrupts those hooks.
        self.pipe.enable_model_cpu_offload()

        # IMPROVEMENT: enable VAE tiling to reduce peak VRAM during decode.
        # At 768×448 × 201 frames the VAE decode alone can spike >8 GB; tiling
        # processes spatial tiles sequentially, capping that spike.
        self.pipe.vae.enable_tiling()
        print(f"✓ {LTX_MODEL} ready")

    @modal.method()
    def generate(
        self,
        first_frame_bytes: bytes,
        last_frame_bytes: bytes,
        prompt: str,
        negative_prompt: str = (
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "watermark, text, signature"
        ),
        width: int = LTX_WIDTH,
        height: int = LTX_HEIGHT,
        num_frames: int = LTX_NUM_FRAMES,
        frame_rate: int = LTX_FRAME_RATE,
        guidance_scale: float = LTX_GUIDANCE,
        steps: int = 50,
        seed: int = 42,
        # IMPROVEMENT: LTX-specific decode parameters for sharper output.
        # decode_timestep controls when in the diffusion schedule the VAE
        # decoder fires; 0.05 is the recommended value for 0.9.5.
        decode_timestep: float = 0.05,
        decode_noise_scale: float = 0.025,
    ) -> bytes:
        """
        Generate an ~8-second video that starts at first_frame and ends at
        last_frame, with intermediate motion guided by `prompt`.

        Uses LTXConditionPipeline with two LTXVideoCondition objects:
          - condition at frame_index=0           pins the first frame
          - condition at frame_index=num_frames-1 pins the last frame

        Returns raw MP4 bytes.
        """
        import os
        import tempfile
        import torch
        from pathlib import Path
        from PIL import Image
        # FIX: correct import — LTXVideoCondition, not LTXVideoConditioningItem,
        # from pipeline_ltx_condition, not pipeline_ltx_image2video.
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import export_to_video

        # Resize inputs to the target resolution (must be divisible by 32).
        first = Image.open(io.BytesIO(first_frame_bytes)).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )
        last = Image.open(io.BytesIO(last_frame_bytes)).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )

        generator = torch.Generator("cpu").manual_seed(seed)

        # Pin frame 0 to first_frame, frame (num_frames−1) to last_frame.
        conditions = [
            LTXVideoCondition(image=first, frame_index=0),
            LTXVideoCondition(image=last,  frame_index=num_frames - 1),
        ]

        output = self.pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            generator=generator,
        )

        # output.frames[0] is a list of PIL Images (length = num_frames).
        frames = output.frames[0]

        # Write to a /tmp file; export_to_video is more reliable than imageio
        # in-memory writes across diffusers versions.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            export_to_video(frames, tmp_path, fps=frame_rate)
            return Path(tmp_path).read_bytes()
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Web endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.function()
@modal.web_endpoint(method="POST")
def image_endpoint(item: dict) -> dict:
    """
    POST  { "prompt": str, "seeds": [42, 1337, 99999],
            "width": 1024, "height": 576,
            "character_ref_b64": "<base64 PNG> | null" }
    →     { "images": ["<base64 PNG>", ...] }
    """
    import base64

    char_bytes = (
        base64.b64decode(item["character_ref_b64"])
        if item.get("character_ref_b64")
        else None
    )
    png_list = FluxGenerator().generate_batch.remote(
        prompt=item["prompt"],
        seeds=item.get("seeds", [42, 1337, 99999]),
        width=item.get("width", 1024),
        height=item.get("height", 576),
        character_ref_bytes=char_bytes,
    )
    return {"images": [base64.b64encode(b).decode() for b in png_list]}


@app.function()
@modal.web_endpoint(method="POST")
def video_endpoint(item: dict) -> dict:
    """
    POST  { "first_frame_b64": "<base64 PNG>",
            "last_frame_b64":  "<base64 PNG>",
            "prompt": str, "fragment_id": int, "seed": 42 }
    →     { "fragment_id": int, "video_b64": "<base64 MP4>" }
    """
    import base64

    first = base64.b64decode(item["first_frame_b64"])
    last  = base64.b64decode(item["last_frame_b64"])

    mp4_bytes = LTXVideoGenerator().generate.remote(
        first_frame_bytes=first,
        last_frame_bytes=last,
        prompt=item.get("prompt", ""),
        seed=item.get("seed", 42),
    )
    return {
        "fragment_id": item.get("fragment_id", -1),
        "video_b64":   base64.b64encode(mp4_bytes).decode(),
    }