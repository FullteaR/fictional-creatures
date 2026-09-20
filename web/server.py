"""架空固有種 観察記録 — ボタンを押すとカードを1枚作る。

生成そのものは src/pipeline.py。ここがやるのは「同時に1件しか走らせない」ことと
「出来上がった1枚を返す」ことだけ。llama-server と comfyui で GPU の 22.2GB が
埋まっているので、2件目は並べても載らない。

生成中のトークンは stdout に出る (pipeline に emit を渡していないため)。
追いたいときは docker compose logs -f webui。
"""

import asyncio
import os
import re
import sys

sys.path.insert(0, os.environ.get("PIPELINE_DIR", "/mnt"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_IMAGE_NAME = re.compile(r"^[^/\\]+\.png$")

# 画面はこれを 2 秒おきに見に来るだけ
_state = {"running": False, "image": None, "error": None}

app = FastAPI(title="架空固有種 観察記録")


async def _generate():
    try:
        card, _ = await asyncio.to_thread(pipeline.generate_card)
        _state["image"] = card["image"]
    except Exception as error:  # 次のボタンで作り直せるよう、理由だけ残して畳む
        _state["error"] = f"{type(error).__name__}: {error}"
        print(f"[generate] failed: {_state['error']}", flush=True)
    finally:
        _state["running"] = False


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/generate")
async def generate():
    if _state["running"]:
        raise HTTPException(status_code=409, detail="すでに観察中です")
    _state.update(running=True, image=None, error=None)
    asyncio.create_task(_generate())
    return {"running": True}


@app.get("/api/status")
async def status():
    return _state


@app.get("/api/image/{image}")
async def image(image: str):
    if not _IMAGE_NAME.match(image):
        raise HTTPException(status_code=404, detail="not found")
    path = os.path.realpath(os.path.join(pipeline.OUT_DIR, image))
    if not path.startswith(os.path.realpath(pipeline.OUT_DIR) + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
