"""Web UI のルート。一度に1枚だけ生成すること、図版を出す先を外に出さないこと。"""
import asyncio
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import pipeline
import server

PNG = b"\x89PNG\r\n\x1a\n"
CARD = "20260921-014220-キングハイド.png"


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client():
    server._state.update(running=False, image=None, error=None)
    with TestClient(server.app) as started:
        yield started
    server._state.update(running=False, image=None, error=None)


def settled(client, timeout=5):
    limit = time.time() + timeout
    while time.time() < limit:
        state = client.get("/api/status").json()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError("生成が終わらない")


def test_status_is_the_three_things_the_page_polls(client):
    assert client.get("/api/status").json() == {"running": False, "image": None, "error": None}


def test_the_page_and_its_assets_are_served(client):
    assert "架空固有種" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_a_card_in_the_directory_is_served(client, out_dir):
    (out_dir / CARD).write_bytes(PNG)
    answer = client.get("/api/image/" + CARD)
    assert answer.status_code == 200
    assert answer.headers["content-type"] == "image/png"
    assert answer.content == PNG


def test_a_card_that_is_not_there_is_a_404(client, out_dir):
    assert client.get("/api/image/" + CARD).status_code == 404


def test_a_name_that_is_not_a_png_is_refused(client, out_dir):
    (out_dir / "notes.txt").write_bytes(b"x")
    assert client.get("/api/image/notes.txt").status_code == 404


def test_a_name_carrying_a_separator_is_refused(client, out_dir):
    (out_dir.parent / "outside.png").write_bytes(PNG)
    assert client.get("/api/image/..%2Foutside.png").status_code == 404
    for name in ("../outside.png", "sub/" + CARD, "sub\\" + CARD):
        with pytest.raises(HTTPException) as refused:
            asyncio.run(server.image(name))
        assert refused.value.status_code == 404


def test_a_link_pointing_out_of_the_directory_is_refused(client, out_dir):
    (out_dir.parent / "outside.png").write_bytes(PNG)
    (out_dir / "linked.png").symlink_to(out_dir.parent / "outside.png")
    assert client.get("/api/image/linked.png").status_code == 404


def test_only_one_card_is_generated_at_a_time(client, monkeypatch):
    release = threading.Event()

    def slow():
        release.wait(5)
        return {"image": CARD}, None

    monkeypatch.setattr(pipeline, "generate_card", slow)
    assert client.post("/api/generate").json() == {"running": True}
    assert client.get("/api/status").json()["running"]
    assert client.post("/api/generate").status_code == 409
    release.set()
    state = settled(client)
    assert state == {"running": False, "image": CARD, "error": None}


def test_a_failed_generation_leaves_its_reason_and_frees_the_button(client, monkeypatch):
    def broken():
        raise RuntimeError("comfyui is down")

    monkeypatch.setattr(pipeline, "generate_card", broken)
    client.post("/api/generate")
    state = settled(client)
    assert state["error"] == "RuntimeError: comfyui is down"
    assert state["image"] is None
    assert client.post("/api/generate").status_code == 200
    settled(client)


def test_pressing_again_clears_what_the_last_card_left(client, monkeypatch):
    monkeypatch.setattr(pipeline, "generate_card", lambda: ({"image": CARD}, None))
    client.post("/api/generate")
    assert settled(client)["image"] == CARD
    release = threading.Event()

    def slow():
        release.wait(5)
        return {"image": "b.png"}, None

    monkeypatch.setattr(pipeline, "generate_card", slow)
    client.post("/api/generate")
    assert client.get("/api/status").json() == {"running": True, "image": None, "error": None}
    release.set()
    assert settled(client)["image"] == "b.png"
