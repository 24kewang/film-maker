"""
pipeline.py
-----------
Orchestrates the full AI film pipeline in three phases:

  Phase 1 (run_phase_1):
    1. Generate script (10 fragments) via Claude
    2. Generate character appearance descriptions via Claude
    3. Generate environment description via Claude
    4. Generate keyframe descriptions via Claude
    5. Generate character + environment reference images (3 variants each)
    → PAUSE: user selects preferred reference variants

  Phase 2 (run_phase_2):
    6. Generate 11 keyframe images (3 variants each) using selected
       character + environment refs as multi-image CLIP conditioning
    → PAUSE: user selects preferred keyframe variants

  Phase 3 (run_phase_3):
    7. Generate 10 video clips via LTX-Video (parallel on Modal)
    8. Assemble final film with ffmpeg

Keyframe model (N+1 boundary frames for N segments)
────────────────────────────────────────────────────
  kf_0  ──[seg 0]── kf_1 ──[seg 1]── kf_2 ── … ── kf_10

  kf_0  = opening frame
  kf_i  = shared boundary between segments i-1 and i
  kf_10 = closing frame

Multi-reference conditioning
────────────────────────────
Each keyframe image is generated with CLIP embeddings from ALL relevant
character refs (based on which characters appear at that boundary) PLUS
the environment ref. These are averaged and blended into FLUX's pooled
text embedding to maintain visual consistency.

Limitation: CLIP embedding averaging is a lightweight approximation.
True multi-subject IP-Adapter conditioning (separate attention injection
per reference) would require deeper pipeline surgery. The averaging
approach provides meaningful consistency but is not perfect for scenes
with many distinct characters.
"""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    """Strip markdown fences and preamble from LLM responses before JSON parsing."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Drop first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODAL_IMAGE_ENDPOINT = os.environ.get(
    "MODAL_IMAGE_ENDPOINT",
    "https://<workspace>--ai-film-pipeline-image-endpoint.modal.run",
)
MODAL_VIDEO_ENDPOINT = os.environ.get(
    "MODAL_VIDEO_ENDPOINT",
    "https://<workspace>--ai-film-pipeline-video-endpoint.modal.run",
)

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

NUM_SEGMENTS  = 10
NUM_KEYFRAMES = NUM_SEGMENTS + 1   # 11


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class ScriptFragment:
    fragment_id: int
    title: str
    narration: str
    action: str
    dialogue: str
    characters_present: list[str]
    environment: str
    duration_seconds: int = 8


@dataclass
class KeyframeDescription:
    keyframe_id: int
    role: str
    prompt: str
    characters_present: list[str] = field(default_factory=list)


@dataclass
class GeneratedImages:
    prompt: str
    variants: list[str]      # base64 PNGs
    chosen_index: int = 0


@dataclass
class PipelineState:
    idea: str = ""
    fragments: list[ScriptFragment] = field(default_factory=list)
    character_names: list[str] = field(default_factory=list)
    environment_name: str = ""

    # NEW: LLM-generated descriptions for richer image prompts
    character_descriptions: dict[str, str] = field(default_factory=dict)
    environment_description: str = ""

    # Reference images (3 variants each)
    character_refs: dict[str, GeneratedImages] = field(default_factory=dict)
    environment_ref: GeneratedImages | None = None

    # 11 boundary keyframe descriptions
    keyframe_descriptions: list[KeyframeDescription] = field(default_factory=list)

    # 11 keyframe images keyed "kf_0" … "kf_10" (3 variants each)
    keyframe_images: dict[str, GeneratedImages] = field(default_factory=dict)

    # Video clips keyed by fragment_id (0-9)
    video_clips: dict[int, bytes] = field(default_factory=dict)

    final_film_path: str = ""


# ---------------------------------------------------------------------------
# Step 1 – Script generation
# ---------------------------------------------------------------------------
SCRIPT_SYSTEM = """\
You are a professional short-film screenwriter.
Given a film idea, produce a tight 1-to-2-minute script divided into
EXACTLY 10 fragments of ~8 seconds each.

CRITICAL RULES FOR VISUAL CONTINUITY:
- The film should use AT MOST 2-3 distinct locations/environments.
- Location changes must be GRADUAL — never cut between completely
  different environments in adjacent segments.
- If a location change is needed, dedicate a transition segment where
  the character moves between locations (e.g. walking out a door,
  driving, a corridor connecting two rooms).
- Prefer stories that can be told in a SINGLE primary location with
  minor setting variations (different angles, lighting shifts, moving
  to an adjacent room).
  
Respond ONLY with a JSON array (no markdown fences, no preamble) where each
element has these keys:
  fragment_id          (integer 0-9)
  title                (short scene label)
  narration            (voiceover / descriptive narration, 1-2 sentences)
  action               (what physically happens on screen)
  dialogue             (spoken lines, or "" if none)
  characters_present   (array of character name strings)
  environment          (brief location description)
  duration_seconds     (always 8)

Keep characters consistent across all fragments (same names and descriptions).
"""


def _generate_script_sync(idea: str) -> list[ScriptFragment]:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SCRIPT_SYSTEM,
        messages=[{"role": "user", "content": idea}],
    )
    data: list[dict] = json.loads(_extract_json(response.content[0].text))
    return [ScriptFragment(**d) for d in data]


async def generate_script(idea: str) -> list[ScriptFragment]:
    return await asyncio.to_thread(_generate_script_sync, idea)


# ---------------------------------------------------------------------------
# Step 2 – Character description generation (NEW)
# ---------------------------------------------------------------------------
CHARACTER_DESC_SYSTEM = """\
You are a character designer for a short film. Given a film script,
produce detailed visual appearance descriptions for each character.

For each character, describe:
  - Age, build, height
  - Hair color, style, length
  - Skin tone
  - Facial features (shape, notable features)
  - Clothing and accessories in detail
  - Overall vibe/energy

These descriptions will be used to generate consistent reference images
with an AI image generator, so be SPECIFIC and VISUAL — avoid abstract
personality traits. Focus on what a camera would capture.

Respond ONLY with a JSON object (no markdown fences, no preamble) mapping
character name → description string.

Example: {"Elena": "Woman in her early 30s, athletic build, shoulder-length
wavy auburn hair, olive skin, sharp green eyes, wearing a weathered brown
leather jacket over a dark turtleneck, fitted cargo pants, and scuffed boots.
A small crescent moon pendant hangs at her collarbone."}
"""


def _generate_character_descriptions_sync(
    idea: str, fragments: list[ScriptFragment]
) -> dict[str, str]:
    script_summary = "\n".join(
        f"Segment {f.fragment_id} — {f.title}: {f.action}. {f.narration}"
        for f in fragments
    )
    characters = list({c for f in fragments for c in f.characters_present})
    user_msg = (
        f"Film idea: {idea}\n\n"
        f"Script:\n{script_summary}\n\n"
        f"Characters to describe: {', '.join(characters)}\n\n"
        f"Generate detailed visual descriptions for each character."
    )
    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=CHARACTER_DESC_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    return json.loads(_extract_json(resp.content[0].text))


async def generate_character_descriptions(
    idea: str, fragments: list[ScriptFragment]
) -> dict[str, str]:
    return await asyncio.to_thread(_generate_character_descriptions_sync, idea, fragments)


# ---------------------------------------------------------------------------
# Step 3 – Environment description generation (NEW)
# ---------------------------------------------------------------------------
ENVIRONMENT_DESC_SYSTEM = """\
You are a production designer for a short film. Given a film script,
produce a detailed visual description of the primary environment/setting.

Describe:
  - Location type and scale
  - Architectural style or natural features
  - Lighting conditions (time of day, light sources, mood)
  - Color palette of the environment
  - Key props, textures, materials
  - Atmospheric details (weather, haze, dust, etc.)
  - Overall cinematic mood

Be SPECIFIC and VISUAL. This will drive AI image generation, so describe
what a camera would see — not abstract moods.

Respond ONLY with a JSON object (no fences, no preamble):
  {"name": "<environment name>", "description": "<detailed description>"}
"""


def _generate_environment_description_sync(
    idea: str, fragments: list[ScriptFragment]
) -> tuple[str, str]:
    script_summary = "\n".join(
        f"Segment {f.fragment_id} — {f.title}: {f.environment}. {f.action}"
        for f in fragments
    )
    environments = list({f.environment for f in fragments})
    user_msg = (
        f"Film idea: {idea}\n\n"
        f"Script:\n{script_summary}\n\n"
        f"Environments mentioned: {', '.join(environments)}\n\n"
        f"Generate a unified description of the primary environment."
    )
    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=ENVIRONMENT_DESC_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    data = json.loads(_extract_json(resp.content[0].text))
    return data["name"], data["description"]


async def generate_environment_description(
    idea: str, fragments: list[ScriptFragment]
) -> tuple[str, str]:
    return await asyncio.to_thread(_generate_environment_description_sync, idea, fragments)


# ---------------------------------------------------------------------------
# Step 4 – Keyframe description generation
# ---------------------------------------------------------------------------
KEYFRAME_SYSTEM = """\
You are a cinematographer and storyboard artist working on a short film.

The film is divided into 10 consecutive 8-second segments. You must produce
EXACTLY 11 boundary-keyframe descriptions — one for each temporal boundary
between segments (plus the opening and closing frames).

Keyframe roles:
  kf_0    "opening"    : The very first frame of the film.
  kf_1-9  "transition" : The shared boundary between two consecutive segments.
                         This frame must make visual sense as BOTH the last
                         frame of segment i-1 AND the first frame of segment i.
  kf_10   "closing"    : The very last frame of the film.

For each keyframe, you are given the character appearance descriptions and
environment description. USE THESE EXACTLY to maintain visual consistency.
Embed the specific clothing, hair, and feature details into each prompt.

For each keyframe write a concise FLUX image-generation prompt of 50–70 words.
Structure it as: [camera angle], [subject/character positions and expressions],
[environment and lighting], [colour palette], [mood]. Do not write full sentences —
use comma-separated phrases. Do not use the word "prompt".

Example format (do not copy content, only structure):
"Wide low-angle shot, Ewan stands at cliff edge arms outstretched, storm lighthouse
behind him, crashing waves below, cold blue-grey palette with amber lamp glow,
mood of desperate hope"

Also specify which characters are present in each keyframe.

CRITICAL: Adjacent keyframes MUST be visually compatible for smooth
video interpolation. Avoid dramatic changes in camera angle, lighting,
or environment between consecutive keyframes. If the story requires a
location change, make it gradual — e.g. kf_i shows a character at a
doorway, kf_{i+1} shows them stepping through.

Respond ONLY with a JSON array of exactly 11 objects (no fences, no preamble):
  keyframe_id          (integer 0-10)
  role                 ("opening" | "transition" | "closing")
  prompt               (detailed image-gen string)
  characters_present   (array of character name strings in this keyframe)
"""


def _generate_keyframe_descriptions_sync(
    fragments: list[ScriptFragment],
    character_descriptions: dict[str, str],
    environment_description: str,
) -> list[KeyframeDescription]:
    script_summary = "\n".join(
        f"Segment {f.fragment_id} — {f.title}: {f.action}. {f.narration}"
        for f in fragments
    )
    char_block = "\n".join(
        f"  {name}: {desc}" for name, desc in character_descriptions.items()
    )
    user_msg = (
        f"Film script (10 segments):\n{script_summary}\n\n"
        f"Character appearances:\n{char_block}\n\n"
        f"Environment: {environment_description}\n\n"
        f"Generate the 11 boundary keyframe descriptions. "
        f"Embed the character appearance details directly into each prompt."
    )
    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        system=KEYFRAME_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    data: list[dict] = json.loads(_extract_json(resp.content[0].text))
    return [KeyframeDescription(**d) for d in data]


async def generate_keyframe_descriptions(
    fragments: list[ScriptFragment],
    character_descriptions: dict[str, str],
    environment_description: str,
) -> list[KeyframeDescription]:
    return await asyncio.to_thread(
        _generate_keyframe_descriptions_sync,
        fragments, character_descriptions, environment_description,
    )


# ---------------------------------------------------------------------------
# Image generation via Modal FLUX endpoint
# ---------------------------------------------------------------------------
async def _call_image_endpoint(
    prompt: str,
    reference_images_b64: list[str] | None = None,
    seeds: list[int] = (42, 1337, 99999),
    width: int = 1024,
    height: int = 576,
) -> GeneratedImages:
    async with httpx.AsyncClient(timeout=360, follow_redirects=True) as client:
        resp = await client.post(
            MODAL_IMAGE_ENDPOINT,
            json={
                "prompt": prompt,
                "seeds": list(seeds),
                "width": width,
                "height": height,
                "reference_images_b64": reference_images_b64,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return GeneratedImages(prompt=prompt, variants=data["images"])


def _get_chosen_ref_b64(gi: GeneratedImages) -> str:
    """Return the chosen variant as base64."""
    return gi.variants[gi.chosen_index]


def _get_keyframe_characters(
    keyframe_id: int, fragments: list[ScriptFragment]
) -> list[str]:
    """Determine which characters appear at a given keyframe boundary."""
    chars = set()
    if keyframe_id == 0:
        chars.update(fragments[0].characters_present)
    elif keyframe_id >= len(fragments):
        chars.update(fragments[-1].characters_present)
    else:
        # Transition: union of characters from adjacent segments
        chars.update(fragments[keyframe_id - 1].characters_present)
        chars.update(fragments[keyframe_id].characters_present)
    return sorted(chars)


async def generate_all_refs(state: PipelineState) -> None:
    """Generate character and environment reference images using LLM descriptions."""
    tasks = {}

    for char_name, char_desc in state.character_descriptions.items():
        prompt = (
            f"Character reference portrait of {char_name}: {char_desc}. "
            f"Full body portrait, cinematic lighting, ultra-detailed, "
            f"neutral background, concept art style, consistent character design"
        )
        tasks[f"char:{char_name}"] = _call_image_endpoint(prompt)

    env_prompt = (
        f"Environment concept art: {state.environment_name}. "
        f"{state.environment_description}. "
        f"Cinematic wide establishing shot, dramatic lighting, ultra-detailed, film still"
    )
    tasks["env"] = _call_image_endpoint(env_prompt)

    results = await asyncio.gather(*tasks.values())
    for key, result in zip(tasks.keys(), results):
        if key.startswith("char:"):
            state.character_refs[key[5:]] = result
        else:
            state.environment_ref = result


async def generate_all_keyframe_images(state: PipelineState) -> None:
    """
    Generate 3 variants for each of the 11 boundary keyframes.

    For each keyframe, assembles a list of reference images:
      - Selected variant for each character present at that boundary
      - Selected environment variant
    These are all passed to FLUX as multi-reference CLIP conditioning.
    """
    style_suffix = (
        "Cinematic film still, 16:9 aspect ratio, photorealistic, 8K resolution, "
        "no watermarks, no text, no UI elements"
    )

    # Collect chosen ref images
    env_b64 = (
        _get_chosen_ref_b64(state.environment_ref)
        if state.environment_ref else None
    )

    async def _gen_one_keyframe(kf: KeyframeDescription):
        # Build reference image list for this keyframe
        ref_list = []

        # Add character refs for characters present in this keyframe
        chars_here = kf.characters_present or _get_keyframe_characters(
            kf.keyframe_id, state.fragments
        )
        for char_name in chars_here:
            if char_name in state.character_refs:
                ref_list.append(
                    _get_chosen_ref_b64(state.character_refs[char_name])
                )

        # Add environment ref
        if env_b64:
            ref_list.append(env_b64)

        return f"kf_{kf.keyframe_id}", await _call_image_endpoint(
            prompt=f"{kf.prompt}. {style_suffix}",
            reference_images_b64=ref_list if ref_list else None,
        )

    tasks = [_gen_one_keyframe(kf) for kf in state.keyframe_descriptions]
    results = await asyncio.gather(*tasks)
    for key, gi in results:
        state.keyframe_images[key] = gi


# ---------------------------------------------------------------------------
# Video generation via Modal LTX-Video endpoint
# ---------------------------------------------------------------------------
async def _generate_one_clip(
    fragment: ScriptFragment,
    first_frame_b64: str,
    last_frame_b64: str,
) -> tuple[int, bytes]:
    prompt = f"{fragment.narration} {fragment.action}".strip()
    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        resp = await client.post(
            MODAL_VIDEO_ENDPOINT,
            json={
                "first_frame_b64": first_frame_b64,
                "last_frame_b64":  last_frame_b64,
                "prompt":          prompt,
                "fragment_id":     fragment.fragment_id,
                "seed":            42 + fragment.fragment_id,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return fragment.fragment_id, base64.b64decode(data["video_b64"])


async def generate_all_video_clips(state: PipelineState) -> None:
    def chosen_b64(key: str) -> str:
        gi = state.keyframe_images[key]
        return gi.variants[gi.chosen_index]

    tasks = [
        _generate_one_clip(
            fragment=frag,
            first_frame_b64=chosen_b64(f"kf_{frag.fragment_id}"),
            last_frame_b64=chosen_b64(f"kf_{frag.fragment_id + 1}"),
        )
        for frag in state.fragments
    ]

    results = await asyncio.gather(*tasks)
    state.video_clips = {fid: clip for fid, clip in results}


# ---------------------------------------------------------------------------
# Assembly with ffmpeg
# ---------------------------------------------------------------------------
def assemble_film(state: PipelineState, output_path: str = "final_film.mp4") -> str:
    with tempfile.TemporaryDirectory() as tmp:
        clip_paths = []
        for i in range(NUM_SEGMENTS):
            p = os.path.join(tmp, f"clip_{i:02d}.mp4")
            Path(p).write_bytes(state.video_clips[i])
            clip_paths.append(p)

        list_file = os.path.join(tmp, "clips.txt")
        with open(list_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_file, "-c", "copy", output_path],
            check=True,
        )

    state.final_film_path = output_path
    return output_path


# ---------------------------------------------------------------------------
# Public API — three-phase pipeline
# ---------------------------------------------------------------------------
async def run_phase_1(idea: str, on_progress=None) -> PipelineState:
    """
    Phase 1: LLM generation + reference images.
    Pauses after ref images so user can select preferred variants.
    """
    state = PipelineState(idea=idea)

    async def emit(step, data=None):
        if on_progress:
            await on_progress(step, data)

    # 1. Script
    await emit("script:start")
    state.fragments = await generate_script(idea)
    state.character_names = sorted({c for f in state.fragments for c in f.characters_present})
    state.environment_name = state.fragments[0].environment
    await emit("script:done", {
        "fragments": [f.__dict__ for f in state.fragments],
        "character_names": state.character_names,
        "environment_name": state.environment_name,
    })

    # 2. Character descriptions
    await emit("character_descriptions:start")
    state.character_descriptions = await generate_character_descriptions(idea, state.fragments)
    await emit("character_descriptions:done", state.character_descriptions)

    # 3. Environment description
    await emit("environment_description:start")
    env_name, env_desc = await generate_environment_description(idea, state.fragments)
    state.environment_name = env_name
    state.environment_description = env_desc
    await emit("environment_description:done", {
        "name": env_name, "description": env_desc,
    })

    # 4. Keyframe descriptions
    await emit("keyframes:start")
    state.keyframe_descriptions = await generate_keyframe_descriptions(
        state.fragments,
        state.character_descriptions,
        state.environment_description,
    )
    await emit("keyframes:done", [kf.__dict__ for kf in state.keyframe_descriptions])

    # 5. Reference images
    await emit("refs:start")
    await generate_all_refs(state)
    await emit("refs:done")

    return state


async def run_phase_2(state: PipelineState, on_progress=None) -> PipelineState:
    """
    Phase 2: Generate keyframe images using user-selected reference variants.
    Pauses after so user can select preferred keyframe variants.
    """
    async def emit(step, data=None):
        if on_progress:
            await on_progress(step, data)

    await emit("keyframe_images:start")
    await generate_all_keyframe_images(state)
    await emit("keyframe_images:done")

    return state


async def run_phase_3(state: PipelineState, on_progress=None) -> PipelineState:
    """Phase 3: Video generation + ffmpeg assembly."""
    async def emit(step, data=None):
        if on_progress:
            await on_progress(step, data)

    await emit("video:start")
    await generate_all_video_clips(state)
    await emit("video:done")

    await emit("assembly:start")
    assemble_film(state)
    await emit("assembly:done", {"path": state.final_film_path})

    return state