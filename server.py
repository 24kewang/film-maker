"""
server.py
---------
FastAPI server. Exposes the pipeline over REST + WebSocket.

Environment variables required:
  ANTHROPIC_API_KEY
  MODAL_IMAGE_ENDPOINT   – from `modal deploy modal_app.py`
  MODAL_VIDEO_ENDPOINT   – from `modal deploy modal_app.py`

Start:  uvicorn server:app --reload --port 8000

Endpoints
─────────
  POST /pipeline/start          kick off steps 1-4, returns { session_id }
  GET  /pipeline/{id}/state     full serialised state (images, fragments…)
  POST /pipeline/{id}/select    save user variant selections
  POST /pipeline/{id}/continue  run steps 5-6 (LTX video + ffmpeg assembly)
  GET  /film/{id}               download final MP4
  WS   /pipeline/{id}/ws        real-time progress events
"""

import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import PipelineState, run_pipeline, continue_pipeline, NUM_KEYFRAMES

app = FastAPI(title="AI Film Pipeline", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory session store (swap for Redis in production)
sessions: dict[str, PipelineState] = {}
ws_connections: dict[str, list[WebSocket]] = {}


# ---------------------------------------------------------------------------
# WebSocket broadcast helper
# ---------------------------------------------------------------------------
async def broadcast(session_id: str, event: str, data: Any = None) -> None:
    msg = json.dumps({"event": event, "data": data})
    for ws in list(ws_connections.get(session_id, [])):
        try:
            await ws.send_text(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    idea: str


class SelectionRequest(BaseModel):
    """
    Map of key → chosen variant index (0-2).

    Keys:
      "char:<name>"   – character reference selection
      "env"           – environment reference selection
      "kf_<n>"        – keyframe image selection (0 … NUM_KEYFRAMES-1)
    """
    selections: dict[str, int]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/pipeline/start")
async def start_pipeline(req: StartRequest):
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = PipelineState(idea=req.idea)
    ws_connections[sid] = []

    async def on_progress(step, data=None):
        await broadcast(sid, step, data)

    asyncio.create_task(_run_steps_1_4(sid, req.idea, on_progress))
    return {"session_id": sid}


async def _run_steps_1_4(sid: str, idea: str, on_progress):
    try:
        state = await run_pipeline(idea, on_progress)
        sessions[sid] = state
        await broadcast(sid, "ready_for_selection")
    except Exception as e:
        await broadcast(sid, "error", {"message": str(e)})


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
        # 11 keyframe image entries, keyed kf_0 … kf_10
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


@app.post("/pipeline/{sid}/select")
async def save_selections(sid: str, req: SelectionRequest):
    """
    Save the user's chosen variant index for any image.
    Accepts mixed keys: "char:Alex", "env", "kf_0" … "kf_10"
    """
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


@app.post("/pipeline/{sid}/continue")
async def continue_with_video(sid: str):
    """Trigger parallel LTX-Video generation + ffmpeg assembly (steps 5-6)."""
    state = sessions.get(sid)
    if not state:
        raise HTTPException(404, "Session not found")

    async def on_progress(step, data=None):
        await broadcast(sid, step, data)

    asyncio.create_task(_run_steps_5_6(sid, state, on_progress))
    return {"ok": True, "message": "LTX-Video generation started"}


async def _run_steps_5_6(sid: str, state: PipelineState, on_progress):
    try:
        await continue_pipeline(state, on_progress)
        sessions[sid] = state
        await broadcast(sid, "film:ready", {"session_id": sid})
    except Exception as e:
        await broadcast(sid, "error", {"message": str(e)})


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
            await websocket.receive_text()    # keep-alive ping/pong
    except WebSocketDisconnect:
        ws_connections[sid].remove(websocket)