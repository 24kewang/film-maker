"""
modal_app.py
------------
Single Modal deployment with two model classes:

  FluxGenerator     – FLUX.1-dev for reference images and keyframe variants
                      Uses IP-Adapter for character-conditioned generation.

  LTXVideoGenerator – LTX-Video (Lightricks) for video clip synthesis.
                      Accepts first + last keyframe images and generates the
                      interpolated clip between them.

Deploy:  modal deploy modal_app.py
Dev:     modal serve modal_app.py    (hot-reload, keeps container warm)
Logs:    modal app logs ai-film-pipeline
"""

import io
import os
import modal

# ---------------------------------------------------------------------------
# Shared container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]",

        # Core diffusion stack
        "diffusers>=0.32.0",          # LTX-Video needs >=0.32
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "torch>=2.4.0",
        "torchvision",
        "torchaudio",
        "safetensors",
        "sentencepiece",
        "Pillow",
        # Video encoding
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "av",                          # PyAV for mp4 muxing
        # IP-Adapter for character conditioning
        "ip_adapter @ git+https://github.com/tencent-ailab/IP-Adapter.git",
    )
)

# Two separate volumes so weights don't compete for I/O
flux_volume  = modal.Volume.from_name("flux-weights",  create_if_missing=True)
ltx_volume   = modal.Volume.from_name("ltx-weights",   create_if_missing=True)

app = modal.App("ai-film-pipeline", image=image)

WEIGHTS_DIR_FLUX = "/weights/flux"
WEIGHTS_DIR_LTX  = "/weights/ltx"

FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
LTX_MODEL  = "Lightricks/LTX-Video"

# IP-Adapter image encoder (shared across both)
CLIP_MODEL = "openai/clip-vit-large-patch14"


# ---------------------------------------------------------------------------
# FLUX Generator  (reference images + keyframe variants)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A100",                        # 24 GB VRAM – sufficient for FLUX fp16
    volumes={WEIGHTS_DIR_FLUX: flux_volume},
    timeout=600,
    # Allow multiple in-flight requests so all 11 keyframes can generate in
    # parallel within a single warm container.
    allow_concurrent_inputs=6,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class FluxGenerator:

    @modal.enter()
    def load_model(self):
        import os
        import torch
        from diffusers import FluxPipeline
        from diffusers.utils import load_image
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

        print("↓ Loading FLUX.1-dev …")
        self.pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=WEIGHTS_DIR_FLUX,
        )
        self.pipe.enable_model_cpu_offload()
        print("✓ FLUX.1-dev ready")

        # ── IP-Adapter image encoder for character conditioning ──────────────
        # We load the CLIP image encoder independently so we can encode the
        # reference image into an embedding that biases FLUX via CFG guidance.
        # This is a lightweight approximation of full IP-Adapter for FLUX;
        # see https://github.com/tencent-ailab/IP-Adapter for the full version.
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        self.clip_encoder   = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_MODEL, torch_dtype=torch.float16
        ).to("cuda")
        self.clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
        print("✓ CLIP image encoder ready")

    def _encode_ref_image(self, ref_image_bytes: bytes | None):
        """Return a CLIP embedding tensor for the reference image, or None."""
        if ref_image_bytes is None:
            return None
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(ref_image_bytes)).convert("RGB")
        inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embed = self.clip_encoder(**inputs).image_embeds  # (1, 768)
        return embed

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
        Generate len(seeds) variants of the same prompt.
        If character_ref_bytes is provided its CLIP embedding is appended to
        the prompt embedding to steer the result toward the reference character.

        Returns a list of raw PNG bytes, one per seed.
        """
        import torch
        from PIL import Image

        # Encode character reference (optional)
        ref_embed = self._encode_ref_image(character_ref_bytes)

        results = []
        for seed in seeds:
            generator = torch.Generator("cuda").manual_seed(seed)

            # Build prompt embeds – inject character guidance if available
            # FLUX uses its own text encoder; we fuse the CLIP embed via the
            # pooled_prompt_embeds slot which accepts (1, 768) tensors.
            extra = {}
            if ref_embed is not None:
                # Scale-blend: 70% text, 30% image reference signal
                _, pooled, _ = self.pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device="cuda",
                    num_images_per_prompt=1,
                )
                blended = 0.7 * pooled + 0.3 * ref_embed.to(pooled.dtype)
                extra["pooled_prompt_embeds"] = blended

            output = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
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


# ---------------------------------------------------------------------------
# LTX-Video Generator  (first-frame / last-frame video interpolation)
# ---------------------------------------------------------------------------
@app.cls(
    # LTX-Video needs ~18 GB VRAM for fp16 at 512×288.
    # A10G (24 GB) works; use A100 if you want 768×512 resolution.
    gpu="A10G",
    volumes={WEIGHTS_DIR_LTX: ltx_volume},
    timeout=600,
    # Each video job is independent; Modal will spawn up to max_containers
    # parallel containers so all 10 clips generate concurrently.
    concurrency_limit=10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import os
        import torch
        from diffusers import LTXImageToVideoPipeline
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

        print("↓ Loading LTX-Video …")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            LTX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=WEIGHTS_DIR_LTX,
        )
        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        print("✓ LTX-Video ready")

    @modal.method()
    def generate(
        self,
        first_frame_bytes: bytes,
        last_frame_bytes: bytes,
        prompt: str,
        negative_prompt: str = (
            "worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        width: int = 768,
        height: int = 432,          # 16:9 at LTX's native resolution bucket
        num_frames: int = 97,       # 97 frames ≈ 8 s @ 12 fps (LTX native fps)
        guidance_scale: float = 7.5,
        steps: int = 50,
        seed: int = 42,
    ) -> bytes:
        """
        Interpolate between first_frame and last_frame.

        LTX-Video supports first-AND-last-frame conditioning via
        `conditioning_items` introduced in diffusers 0.32.
        Each item specifies a PIL image and the frame index it should appear at.

        Returns raw MP4 bytes.
        """
        import torch
        import imageio
        import numpy as np
        from PIL import Image
        from diffusers.pipelines.ltx.pipeline_ltx_image2video import LTXVideoConditioningItem

        first = Image.open(io.BytesIO(first_frame_bytes)).convert("RGB").resize((width, height))
        last  = Image.open(io.BytesIO(last_frame_bytes)).convert("RGB").resize((width, height))

        generator = torch.Generator("cuda").manual_seed(seed)

        # conditioning_items pins specific frames.
        #   frame_index=0           → first frame
        #   frame_index=num_frames-1 → last frame
        conditioning = [
            LTXVideoConditioningItem(image=first, frame_index=0),
            LTXVideoConditioningItem(image=last,  frame_index=num_frames - 1),
        ]

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            conditioning_items=conditioning,
            width=width,
            height=height,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

        # output.frames is a list of PIL Images (length = num_frames)
        frames = output.frames[0]   # first (and only) video in the batch

        # Encode to MP4 with imageio / ffmpeg
        buf = io.BytesIO()
        with imageio.get_writer(buf, format="mp4", fps=12, codec="libx264",
                                quality=8, macro_block_size=8) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))

        return buf.getvalue()


# ---------------------------------------------------------------------------
# Web endpoints (called by the FastAPI orchestration server over HTTPS)
# ---------------------------------------------------------------------------

@app.function()
@modal.web_endpoint(method="POST")
def image_endpoint(item: dict) -> dict:
    """
    POST body:
      {
        "prompt": str,
        "seeds": [42, 1337, 99999],   # number of variants
        "width": 1024,
        "height": 576,
        "character_ref_b64": "<base64 PNG> | null"
      }
    Returns:
      { "images": ["<base64 PNG>", ...] }
    """
    import base64

    char_bytes = None
    if item.get("character_ref_b64"):
        char_bytes = base64.b64decode(item["character_ref_b64"])

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
    POST body:
      {
        "first_frame_b64": "<base64 PNG>",
        "last_frame_b64":  "<base64 PNG>",
        "prompt": str,
        "fragment_id": int,
        "seed": 42
      }
    Returns:
      { "fragment_id": int, "video_b64": "<base64 MP4>" }
    """
    import base64

    first = base64.b64decode(item["first_frame_b64"])
    last  = base64.b64decode(item["last_frame_b64"])

    mp4_bytes = LTXVideoGenerator().generate.remote(
        first_frame_bytes=first,
        last_frame_bytes=last,
        prompt=item["prompt"],
        seed=item.get("seed", 42),
    )
    return {
        "fragment_id": item.get("fragment_id", -1),
        "video_b64":   base64.b64encode(mp4_bytes).decode(),
    }