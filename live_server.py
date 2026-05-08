#!/usr/bin/env python3
"""
Live GBP Calibration Server
Bidirectional WebSocket connection between the robot simulator and the GBP solver.

Protocol
--------
Client → Server  (JSON):
  { "type": "pose",  "limbs": [...], "connections": [...] }
  { "type": "reset" }

Server → Client  (JSON):
  {
    "type": "calibration",
    "calibrations": [
      { "id": "calib2", "mean": [x, y], "cov_xy": [[cxx,cxy],[cyx,cyy]] },
      ...
    ],
    "step": <int>,      # number of poses ingested so far
    "energy": <float>   # current GBP energy
  }
  { "type": "error", "message": "..." }

Usage
-----
  python live_server.py
  # or for auto-reload during development:
  uvicorn live_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import sys
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# ---------------------------------------------------------------------------
# Import GBP solver components
# ---------------------------------------------------------------------------
# Ensure we can import sibling modules regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtsam_implementation import create_gbp_solver, update_factor_graph, extract_calibrations
# from gbp_implementation import create_gbp_solver, update_factor_graph, extract_calibrations

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Live GBP Calibration Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory that contains robot_simulator.html
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

GBP_ITERS_PER_STEP = 8   # fixed number of GBP iterations per pose step


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Simple health-check so the simulator can detect the server on startup."""
    return {"status": "ok", "server": "live_gbp"}


@app.get("/")
@app.get("/robot_simulator.html")
def serve_simulator():
    """Serve the simulator HTML directly from the same directory."""
    path = os.path.join(STATIC_DIR, "robot_simulator.html")
    return FileResponse(path, media_type="text/html")


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    One WebSocket connection = one calibration session.
    A fresh factor graph is created when the connection opens or when a
    'reset' message is received.
    """
    await websocket.accept()
    log.info("WebSocket client connected – new calibration session started")

    fg = create_gbp_solver()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Invalid JSON: {exc}"
                }))
                continue

            msg_type = msg.get("type")

            # ------------------------------------------------------------------
            # RESET – discard factor graph and start fresh
            # ------------------------------------------------------------------
            if msg_type == "reset":
                fg = create_gbp_solver()
                log.info("Factor graph reset by client")
                await websocket.send_text(json.dumps({
                    "type": "calibration",
                    "calibrations": [],
                    "step": 0,
                    "energy": 0.0
                }))
                continue

            # ------------------------------------------------------------------
            # POSE – ingest new observation, run GBP, return calibration update
            # ------------------------------------------------------------------
            if msg_type == "pose":
                # Validate minimum required fields
                if "limbs" not in msg or "connections" not in msg:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "pose message missing 'limbs' or 'connections'"
                    }))
                    continue

                try:
                    # 1. Add new pose observation to the factor graph
                    update_factor_graph(msg, fg)
                    log.info(
                        "Pose %d ingested – %d var nodes, %d factors",
                        fg.step_count,
                        sum(len(v) for v in fg.var_nodes.values()),
                        len(fg.factors),
                    )

                    # 2. Run fixed number of GBP iterations
                    fg.gbp_solve(n_iters=GBP_ITERS_PER_STEP)

                    # 3. Extract calibration estimates
                    calibrations = extract_calibrations(fg)

                    # 4. Compute current energy
                    try:
                        energy = float(fg.energy())
                    except Exception:
                        energy = 0.0

                    # 5. Send calibration back to the simulator
                    response = {
                        "type": "calibration",
                        "calibrations": calibrations,
                        "step": fg.step_count,
                        "energy": energy,
                    }
                    await websocket.send_text(json.dumps(response))
                    log.info(
                        "Calibration sent – step=%d, energy=%.4f, nodes=%s",
                        fg.step_count,
                        energy,
                        [(c["id"], [round(v, 2) for v in c["mean"]]) for c in calibrations],
                    )

                except Exception as exc:
                    log.exception("Error processing pose: %s", exc)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(exc)
                    }))
                continue

            # ------------------------------------------------------------------
            # Unknown message type
            # ------------------------------------------------------------------
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unknown message type: '{msg_type}'"
            }))

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as exc:
        log.exception("Unexpected WebSocket error: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Live GBP Calibration Server")
    print("=" * 55)
    print("  HTTP  : http://localhost:8000")
    print("  WS    : ws://localhost:8000/ws")
    print("  Health: http://localhost:8000/api/health")
    print()
    print("  Open robot_simulator.html in your browser,")
    print("  or navigate to http://localhost:8000/")
    print("=" * 55)

    # Change to script directory so relative imports/file serving work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    uvicorn.run(
        "live_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
