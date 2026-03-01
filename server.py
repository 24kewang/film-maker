"""
server.py
---------
FastAPI server with three-phase pipeline:

  POST /pipeline/start              Phase 1: script + descriptions + ref images
  POST /pipeline/{id}/select        Save user variant selections (refs or keyframes)
  POST /pipeline/{id}/gen-keyframes Phase 2: keyframe images (uses selected refs)
  POST /pipeline/{id}/continue      Phase 3: LTX video + ffmpeg assembly
  GET  /pipeline/{id}/state         Full serialised state
  GET  /film/{id}                   Download final MP4
  WS   /pipeline/{id}/ws            Real-time progress events

Start:  uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import (
    PipelineState, run_phase_1, run_phase_2, run_phase_3, NUM_KEYFRAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

app = FastAPI(title="AI Film Pipeline", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict[str, PipelineState] = {}
ws_connections: dict[str, list[WebSocket]] = {}


async def broadcast(session_id: str, event: str, data: Any = None) -> None:
    msg = json.dumps({"event": event, "data": data})
    for ws in list(ws_connections.get(session_id, [])):
        try:
            await ws.send_text(msg)
        except Exception:
            pass


class StartRequest(BaseModel):
    idea: str


class SelectionRequest(BaseModel):
    selections: dict[str, int]


# ---------------------------------------------------------------------------
# Phase 1: Script + descriptions + reference images
# ---------------------------------------------------------------------------
@app.post("/pipeline/start")
async def start_pipeline(req: StartRequest):
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = PipelineState(idea=req.idea)
    ws_connections[sid] = []

    async def on_progress(step, data=None):
        await broadcast(sid, step, data)

    asyncio.create_task(_run_phase_1(sid, req.idea, on_progress))
    return {"session_id": sid}


async def _run_phase_1(sid: str, idea: str, on_progress):
    try:
        logger.info(f"[{sid}] Phase 1 starting")
        state = await run_phase_1(idea, on_progress)
        sessions[sid] = state
        logger.info(f"[{sid}] Phase 1 complete — ready for ref selection")
        await broadcast(sid, "ready_for_ref_selection")
    except Exception as e:
        logger.exception(f"[{sid}] Phase 1 failed")
        await broadcast(sid, "error", {"message": str(e)})


# ---------------------------------------------------------------------------
# Save selections (used for both refs and keyframes)
# ---------------------------------------------------------------------------
@app.post("/pipeline/{sid}/select")
async def save_selections(sid: str, req: SelectionRequest):
    state = sessions.get(sid)
    if not state:
        raise HTTPException(404, "Session not found")

    for key, idx in req.selections.items():
        if key.startswith("char:") and key[5:] in state.character_refs:
            state.character_refs[key[5:]].chosen_index = idx
        elif key == "env" and state.environment_ref:
            state.environment_ref.chosen_index = idx
        elif key in state.keyframe_images:
            state.keyframe_images[key].chosen_index = idx

    return {"ok": True}


# ---------------------------------------------------------------------------
# Phase 2: Keyframe image generation (after user selects refs)
# ---------------------------------------------------------------------------
@app.post("/pipeline/{sid}/gen-keyframes")
async def generate_keyframes(sid: str):
    state = sessions.get(sid)
    if not state:
        raise HTTPException(404, "Session not found")

    async def on_progress(step, data=None):
        await broadcast(sid, step, data)

    asyncio.create_task(_run_phase_2(sid, state, on_progress))
    return {"ok": True, "message": "Keyframe image generation started"}


async def _run_phase_2(sid: str, state: PipelineState, on_progress):
    try:
        logger.info(f"[{sid}] Phase 2 starting — keyframe image generation")
        await run_phase_2(state, on_progress)
        sessions[sid] = state
        logger.info(f"[{sid}] Phase 2 complete — ready for keyframe selection")
        await broadcast(sid, "ready_for_keyframe_selection")
    except Exception as e:
        logger.exception(f"[{sid}] Phase 2 failed")
        await broadcast(sid, "error", {"message": str(e)})


# ---------------------------------------------------------------------------
# Phase 3: Video generation + assembly (after user selects keyframes)
# ---------------------------------------------------------------------------
@app.post("/pipeline/{sid}/continue")
async def continue_with_video(sid: str):
    state = sessions.get(sid)
    if not state:
        raise HTTPException(404, "Session not found")

    async def on_progress(step, data=None):
        await broadcast(sid, step, data)

    asyncio.create_task(_run_phase_3(sid, state, on_progress))
    return {"ok": True, "message": "Video generation started"}


async def _run_phase_3(sid: str, state: PipelineState, on_progress):
    try:
        logger.info(f"[{sid}] Phase 3 starting — video generation + assembly")
        await run_phase_3(state, on_progress)
        sessions[sid] = state
        logger.info(f"[{sid}] Phase 3 complete — film ready")
        await broadcast(sid, "film:ready", {"session_id": sid})
    except Exception as e:
        logger.exception(f"[{sid}] Phase 3 failed")
        await broadcast(sid, "error", {"message": str(e)})


# ---------------------------------------------------------------------------
# State + film download
# ---------------------------------------------------------------------------
@app.get("/pipeline/{sid}/state")
async def get_state(sid: str):
    state = sessions.get(sid)
    if not state:
        raise HTTPException(404, "Session not found")

    return {
        "idea": state.idea,
        "fragments": [f.__dict__ for f in state.fragments],
        "character_names": state.character_names,
        "environment_name": state.environment_name,
        "character_descriptions": state.character_descriptions,
        "environment_description": state.environment_description,
        "character_refs": {
            name: {
                "prompt": gi.prompt,
                "variants": gi.variants,
                "chosen_index": gi.chosen_index,
            }
            for name, gi in state.character_refs.items()
        },
        "environment_ref": (
            {
                "prompt": state.environment_ref.prompt,
                "variants": state.environment_ref.variants,
                "chosen_index": state.environment_ref.chosen_index,
            }
            if state.environment_ref else None
        ),
        "keyframe_descriptions": [kf.__dict__ for kf in state.keyframe_descriptions],
        "keyframe_images": {
            key: {
                "prompt": gi.prompt,
                "variants": gi.variants,
                "chosen_index": gi.chosen_index,
            }
            for key, gi in state.keyframe_images.items()
        },
        "num_keyframes": NUM_KEYFRAMES,
        "video_clips_ready": sorted(state.video_clips.keys()),
        "has_final_film": bool(state.final_film_path),
    }


@app.get("/film/{sid}")
async def download_film(sid: str):
    state = sessions.get(sid)
    if not state or not state.final_film_path:
        raise HTTPException(404, "Film not ready yet")
    return FileResponse(state.final_film_path, media_type="video/mp4", filename="film.mp4")


@app.websocket("/pipeline/{sid}/ws")
async def ws_endpoint(websocket: WebSocket, sid: str):
    await websocket.accept()
    ws_connections.setdefault(sid, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_connections[sid].remove(websocket)