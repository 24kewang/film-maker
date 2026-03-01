# Film Stitch

An AI-powered short film generation pipeline that turns text ideas into complete films in minutes.

## Stack

- **Backend:** FastAPI (Python 3.11+) + Uvicorn
- **AI Models:** 
  - Claude 3 (Anthropic) — script & narrative generation
  - FLUX.1-dev — AI image generation (via Modal)
  - LTX-Video-0.9.5 — video interpolation (via Modal)
- **Orchestration:** Modal (GPU inference)
- **Frontend:** React + Vite
- **Video Assembly:** FFmpeg

## Key Features

- **3-Phase Pipeline:** Script generation → Keyframe images → Video synthesis
- **Multi-Reference Conditioning:** Character & environment consistency via CLIP embedding blending
- **Interactive UI:** Real-time variant selection for characters, environments, and keyframes
- **WebSocket Events:** Live progress streaming during generation
- **Parallel Processing:** Concurrent video clip generation on Modal GPUs

## How It Works

**Phase 1 – Script & Assets**
- Claude generates a 10-fragment screenplay from your idea
- Generates character descriptions, environment details, and keyframe prompts
- Creates 3 variant reference images for characters and environment

**Phase 2 – Keyframe Images** (after you select refs)
- Generates 11 keyframe images (3 variants each) using FLUX.1-dev
- Each keyframe uses multi-image CLIP conditioning from selected character/environment refs
- Maintains visual consistency across the film

**Phase 3 – Video Generation & Assembly** (after you select keyframes)
- Generates 10 video clips via LTX-Video (interpolates between keyframes)
- Stitches clips together with ffmpeg into a final MP4

## Quick Setup

### Prerequisites
- Python 3.11+, Node.js 18+, ffmpeg, Git
- Anthropic API key (claude.ai)
- Modal account (modal.com) with HF token for FLUX.1-dev access

### Installation

```bash
# 1. Set up backend
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt

# 2. Create .env file
```

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-your_api_key_here
MODAL_IMAGE_ENDPOINT=https://<workspace>--ai-film-pipeline-image-endpoint.modal.run
MODAL_VIDEO_ENDPOINT=https://<workspace>--ai-film-pipeline-video-endpoint.modal.run
```

Then continue:

```bash
# 3. Set up Modal
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here

# 4. Deploy to Modal
modal deploy modal_app.py

# 5. Start FastAPI server
uvicorn server:app --reload --port 8000

# 6. Start frontend (in new terminal)
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` (Vite default) and describe your film idea.

## What You Get

✨ A complete short film (~2 minutes) generated from a single text prompt, with:
- Original screenplay adapted from your idea
- AI-generated character appearances and environments
- Smooth video transitions between scenes
- Final MP4 ready to download and share

---

**Status:** Ready for local development. Built for experimentation with AI film generation.