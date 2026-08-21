"""ComfyUI 経由の画像生成が通るかを確認するスモークテスト。

    docker compose exec app python /mnt/smoke_test_comfyui.py
    # ホストから直接叩く場合:
    COMFYUI_URL=http://localhost:8188 python src/smoke_test_comfyui.py
"""
import json
import sys
import time
import urllib.request

import imageGenerateUtils as ig


def object_info(node):
    with urllib.request.urlopen(f"{ig.COMFYUI_URL}/object_info/{node}", timeout=30) as res:
        return json.loads(res.read())[node]


def main():
    print(f"ComfyUI: {ig.COMFYUI_URL}")

    ok = True
    for node, field, want in [("UNETLoader", "unet_name", ig.DIFFUSION_MODEL),
                              ("CLIPLoader", "clip_name", ig.TEXT_ENCODER),
                              ("VAELoader", "vae_name", ig.VAE)]:
        found = object_info(node)["input"]["required"][field][0]
        hit = want in found
        ok &= hit
        print(f"  {node:<12} {want:<34} {'OK' if hit else 'MISSING (available: %s)' % found}")

    ks = object_info("KSampler")["input"]["required"]
    samplers, schedulers = ks["sampler_name"][0], ks["scheduler"][0]
    for label, want, pool in [("sampler", ig.SAMPLER, samplers), ("scheduler", ig.SCHEDULER, schedulers)]:
        hit = want in pool
        ok &= hit
        print(f"  {label:<12} {want:<34} {'OK' if hit else 'MISSING (available: %s)' % pool}")

    if not ok:
        print("\n必要なモデル/設定が揃っていません。")
        return 1

    prompt = ("masterpiece, best quality, absurdres, a bioluminescent deep-sea creature, "
              "translucent fins, drifting through dark abyssal water, faint blue glow, "
              "scientific field-guide illustration")
    print("\ngenerating...")
    t0 = time.time()
    img = ig.get_image(prompt, seed=1234)
    print(f"done in {time.time() - t0:.1f}s -> size={img.size} mode={img.mode}")

    out = "/mnt/smoke_test_comfyui.png"
    try:
        img.save(out)
    except OSError:
        out = "smoke_test_comfyui.png"
        img.save(out)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
