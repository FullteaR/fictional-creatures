import io
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

from janome.tokenizer import Tokenizer
from PIL import Image, ImageDraw

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://comfyui:8188").rstrip("/")

# Anima は拡散モデル / テキストエンコーダ / VAE が別ファイルに分かれている
DIFFUSION_MODEL = os.environ.get("COMFYUI_DIFFUSION_MODEL", "novaAnimeAM_v40.safetensors")
TEXT_ENCODER = os.environ.get("COMFYUI_TEXT_ENCODER", "qwen_3_06b_base.safetensors")
VAE = os.environ.get("COMFYUI_VAE", "qwen_image_vae.safetensors")

# 画風はコード側で固定する。LLM に任せると毎回ぶれる上、放っておくと
# masterpiece / 8k / cinematic 系の「盛る」タグを足してきて AI 絵になる。
STYLE_PREFIX = (
    "naturalist field guide illustration, unlabeled, flat illustration, matte finish, "
    "muted limited palette, soft even diffuse light, low contrast, no glare, "
    "clear readable silhouette, legible anatomical structure, "
    "restrained composition, quiet and understated"
)

NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, "
    # ここから下が「AI 絵っぽさ」の除去。演出・発光・過剰レンダリングを潰す
    "glowing, neon, bloom, lens flare, god rays, volumetric lighting, hdr, "
    "oversaturated, high contrast, dramatic lighting, cinematic, vignette, "
    "airbrushed, plastic sheen, glossy, wet look, 3d render, photorealistic, "
    "hyperdetailed, busy background, "
    # 図版スタイルは「解説文が刷り込まれたページ」を呼び込むので、文字類を明示的に潰す
    "text, letters, words, caption, label, title, typography, handwriting, "
    "watermark, signature, logo, page number, printed page, book page"
)

# Anima の対応解像度は 512〜1536px。5:3 で生成してから 800x480 に縮小する
GEN_WIDTH, GEN_HEIGHT = 1280, 768
OUT_WIDTH, OUT_HEIGHT = 800, 480

# Nova Anime AM v4.0 の作者推奨は Euler a / Normal / steps 20-30 / CFG 4-6。
# Anima 公式テンプレートの既定は euler / simple なので、そちらに戻すこともできる。
SAMPLER = os.environ.get("COMFYUI_SAMPLER", "euler_ancestral")
SCHEDULER = os.environ.get("COMFYUI_SCHEDULER", "normal")
STEPS = int(os.environ.get("COMFYUI_STEPS", "30"))
CFG = float(os.environ.get("COMFYUI_CFG", "5.0"))


def _build_workflow(prompt, negative_prompt, width, height, seed, steps, cfg):
    """Anima 公式テンプレート相当のワークフローを ComfyUI の API 形式で組み立てる"""
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": TEXT_ENCODER, "type": "stable_diffusion", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "positive": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": negative_prompt, "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sampler": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
            "model": ["unet", 0], "positive": ["positive", 0],
            "negative": ["negative", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"filename_prefix": "endemic", "images": ["decode", 0]}},
    }


def _request(path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"{COMFYUI_URL}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read())
    except urllib.error.HTTPError as e:
        # ComfyUI はノード検証エラーの詳細をレスポンスボディに入れて返す
        raise RuntimeError(f"ComfyUI {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from None


def get_image(prompt, negative_prompt=NEGATIVE_PROMPT, width=GEN_WIDTH, height=GEN_HEIGHT,
              seed=None, steps=STEPS, cfg=CFG, timeout=600):
    """ComfyUI サーバーに HTTP 経由で生成を依頼し、PIL Image を返す"""
    if seed is None:
        seed = random.randint(0, 2 ** 63 - 1)

    workflow = _build_workflow(f"{STYLE_PREFIX}, {prompt}", negative_prompt, width, height, seed, steps, cfg)
    prompt_id = _request("/prompt", {"prompt": workflow})["prompt_id"]

    deadline = time.time() + timeout
    images = []
    while not images:
        entry = _request(f"/history/{prompt_id}").get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {json.dumps(status, ensure_ascii=False)[:2000]}")
            images = [img for out in entry.get("outputs", {}).values()
                      for img in out.get("images", [])]
        if not images:
            if time.time() > deadline:
                raise TimeoutError(f"ComfyUI did not return an image within {timeout}s")
            time.sleep(1)

    query = urllib.parse.urlencode({
        "filename": images[0]["filename"],
        "subfolder": images[0].get("subfolder", ""),
        "type": images[0].get("type", "output"),
    })
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}", timeout=120) as res:
        image = Image.open(io.BytesIO(res.read()))
        image.load()

    if image.size != (OUT_WIDTH, OUT_HEIGHT):
        image = image.resize((OUT_WIDTH, OUT_HEIGHT), Image.LANCZOS)
    return image.convert("RGB")


def getTextWidth(text, font):
    return font.getbbox(text)[2] - font.getbbox(text)[0]


def getTextHeight(text, font):
    return font.getbbox(text)[3] - font.getbbox(text)[1]


def getLineBreak(text, font, max_width):
    jp_tokenizer = Tokenizer()
    tokens = list(jp_tokenizer.tokenize(text))

    chunks = []
    chunk = ""
    for i, token in enumerate(tokens):
        word = token.surface
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech in ("名詞"):
            if chunk:
                chunks.append(chunk)
            chunk = word
        else:
            chunk += word
    if chunk:
        chunks.append(chunk)

    lines = []
    line = ""
    for chunk in chunks:
        test_line = line+chunk
        if getTextWidth(test_line, font) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = chunk
    if line:
        lines.append(line)
    return lines


def add_caption(name, description, scientific_name, image, title_font, paragraph_font, scientific_font):
    scientific_name = f" (学名: {scientific_name})"
    description = description.replace("\n", "")

    max_width = 420
    lines = getLineBreak(description, paragraph_font, max_width)

    title_height = getTextHeight(name, title_font) + 7
    line_height = getTextHeight("あ", paragraph_font) + 7
    scientific_height = getTextHeight(scientific_name, scientific_font) + 7
    total_height = line_height * len(lines) + title_height
    max_line_width = max([getTextWidth(l, paragraph_font) for l in lines] + [getTextWidth(name, title_font)+getTextWidth(scientific_name, scientific_font)])

    im_w, im_h = image.size
    padding = 10

    def place(near_lo, near_hi, far_lo, far_hi, limit):
        """手前寄せ / 奥寄せをランダムに選ぶ。想定より長いテキストが来ても
        描画開始位置が画像の外に出ないようクランプする"""
        pos = random.choice([random.randint(near_lo, near_hi), random.randint(far_lo, far_hi)])
        return max(padding, min(pos, limit)) if limit >= padding else padding

    x = place(20, 70, im_w-max_line_width-70, im_w-max_line_width-20, im_w-max_line_width-padding)
    y = place(30, 80, im_h-total_height-60, im_h-total_height-10, im_h-total_height-padding)
    bg_color = (0, 0, 0, 128)
    background_box = (x - padding, y - padding,
                          x + max_line_width + padding,
                          y + total_height + padding)
    text_color = "white"


    txt_layer = Image.new('RGBA', image.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt_layer)

    draw.rounded_rectangle(background_box, radius=10, fill=bg_color)

    draw.text((x,y), name, font=title_font, fill=text_color)
    draw.text((x+getTextWidth(name, title_font), y + title_height - scientific_height), scientific_name, font=scientific_font, fill=text_color)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height + title_height), line, font=paragraph_font, fill=text_color)

    return Image.alpha_composite(image.convert('RGBA'), txt_layer).convert("RGB")
