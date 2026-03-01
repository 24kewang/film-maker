"""
modal_app.py  —  AI Film Pipeline
──────────────────────────────────
Two Modal classes:

  FluxGenerator       FLUX.1-dev  →  reference images + keyframe variants
                      Multi-reference CLIP embedding blend for character +
                      environment consistency across the film.
                      GPU: A100-40GB

  LTXVideoGenerator   LTX-Video-0.9.5  →  first-frame / last-frame interpolation
                      Uses LTXConditionPipeline + LTXVideoCondition (diffusers ≥ 0.33)
                      GPU: A100-40GB

Deploy:   modal deploy modal_app.py
Serve:    modal serve  modal_app.py
"""

import io
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "diffusers>=0.33.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "torch>=2.4.0",
        "torchvision",
        "torchaudio",
        "safetensors",
        "sentencepiece",
        "Pillow",
        "imageio[ffmpeg]",
        "imageio-ffmpeg",
        "av",
        "ip-adapter",
    )
)

flux_volume = modal.Volume.from_name("flux-weights", create_if_missing=True)
ltx_volume  = modal.Volume.from_name("ltx-weights",  create_if_missing=True)

app = modal.App("ai-film-pipeline", image=image)

WEIGHTS_DIR_FLUX = "/weights/flux"
WEIGHTS_DIR_LTX  = "/weights/ltx"

FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
LTX_MODEL  = "Lightricks/LTX-Video-0.9.5"
CLIP_MODEL = "openai/clip-vit-large-patch14"

LTX_WIDTH      = 768
LTX_HEIGHT     = 448
LTX_NUM_FRAMES = 201
LTX_FRAME_RATE = 25
LTX_GUIDANCE   = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# FLUX Generator
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="A100-40GB",
    volumes={WEIGHTS_DIR_FLUX: flux_volume},
    timeout=600,
    # FIX: set to 1 to avoid "RuntimeError: Already borrowed" from concurrent
    # tokenizer access. Modal scales horizontally instead.
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
        print("✓ FLUX.1-dev ready")

        print("↓ Loading CLIP image encoder …")
        self.clip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_MODEL, torch_dtype=torch.float16
        ).to("cuda")
        self.clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
        print("✓ CLIP encoder ready")

    def _encode_ref_image(self, ref_image_bytes: bytes):
        """Encode a single reference image to a CLIP embedding."""
        import torch
        from PIL import Image

        img    = Image.open(io.BytesIO(ref_image_bytes)).convert("RGB")
        inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return self.clip_encoder(**inputs).image_embeds  # (1, 768)

    def _encode_multiple_refs(self, ref_images_bytes: list[bytes]):
        """
        Encode multiple reference images and return their averaged CLIP embedding.
        This provides multi-character + environment conditioning in a single vector.
        """
        import torch

        if not ref_images_bytes:
            return None

        embeddings = []
        for ref_bytes in ref_images_bytes:
            emb = self._encode_ref_image(ref_bytes)
            embeddings.append(emb)

        # Average all reference embeddings
        stacked = torch.stack(embeddings, dim=0)  # (N, 1, 768)
        return stacked.mean(dim=0)                 # (1, 768)

    @modal.method()
    def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        width: int = 1024,
        height: int = 576,
        steps: int = 28,
        guidance: float = 3.5,
        reference_images_bytes: list[bytes] | None = None,
        ref_blend_strength: float = 0.3,
    ) -> list[bytes]:
        """
        Generate len(seeds) variants of `prompt`.

        If reference_images_bytes is provided, their CLIP embeddings are averaged
        and blended into FLUX's pooled text embedding at `ref_blend_strength`
        to nudge results toward the reference appearance(s).

        This supports multi-character + environment conditioning — pass all
        relevant reference images for a given keyframe.
        """
        import torch

        ref_embed = self._encode_multiple_refs(reference_images_bytes or [])
        results   = []

        for seed in seeds:
            generator = torch.Generator("cuda").manual_seed(seed)
            extra = {}
            if ref_embed is not None:
                _, pooled, _ = self.pipe.encode_prompt(
                    prompt=prompt, prompt_2=None,
                    device="cuda", num_images_per_prompt=1,
                )
                blend = 1.0 - ref_blend_strength
                extra["pooled_prompt_embeds"] = (
                    blend * pooled + ref_blend_strength * ref_embed.to(pooled.dtype)
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
# LTX-Video Generator
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="A100-40GB",
    volumes={WEIGHTS_DIR_LTX: ltx_volume},
    timeout=900,
    concurrency_limit=10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LTXVideoGenerator:

    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import LTXConditionPipeline
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

        print(f"↓ Loading {LTX_MODEL} …")
        self.pipe = LTXConditionPipeline.from_pretrained(
            LTX_MODEL,
            torch_dtype=torch.bfloat16,
            cache_dir=WEIGHTS_DIR_LTX,
        )
        self.pipe.enable_model_cpu_offload()
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
        decode_timestep: float = 0.05,
        decode_noise_scale: float = 0.025,
    ) -> bytes:
        import os
        import tempfile
        import torch
        from pathlib import Path
        from PIL import Image
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import export_to_video

        first = Image.open(io.BytesIO(first_frame_bytes)).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )
        last = Image.open(io.BytesIO(last_frame_bytes)).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )

        generator = torch.Generator("cpu").manual_seed(seed)

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

        frames = output.frames[0]

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
            "reference_images_b64": ["<base64 PNG>", ...] | null }
    →     { "images": ["<base64 PNG>", ...] }
    """
    import base64

    ref_bytes_list = None
    if item.get("reference_images_b64"):
        ref_bytes_list = [
            base64.b64decode(b64) for b64 in item["reference_images_b64"]
        ]

    png_list = FluxGenerator().generate_batch.remote(
        prompt=item["prompt"],
        seeds=item.get("seeds", [42, 1337, 99999]),
        width=item.get("width", 1024),
        height=item.get("height", 576),
        reference_images_bytes=ref_bytes_list,
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