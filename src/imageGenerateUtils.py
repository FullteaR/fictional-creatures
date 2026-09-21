import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from janome.tokenizer import Tokenizer
from PIL import Image, ImageDraw

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://comfyui:8188").rstrip("/")

DIFFUSION_MODEL = os.environ.get("COMFYUI_DIFFUSION_MODEL", "novaAnimeAM_v40.safetensors")
TEXT_ENCODER = os.environ.get("COMFYUI_TEXT_ENCODER", "qwen_3_06b_base.safetensors")
VAE = os.environ.get("COMFYUI_VAE", "qwen_image_vae.safetensors")

STYLE_PREFIX = (
    "naturalist field guide illustration, unlabeled, flat illustration, matte finish, "
    "muted limited palette, soft even diffuse light, low contrast, no glare, "
    "clear readable silhouette, legible anatomical structure, "
    "restrained composition, quiet and understated"
)

NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, "
    "glowing, neon, bloom, lens flare, god rays, volumetric lighting, hdr, "
    "oversaturated, high contrast, dramatic lighting, cinematic, vignette, "
    "airbrushed, plastic sheen, glossy, wet look, 3d render, photorealistic, "
    "hyperdetailed, busy background, "
    "extra limbs, extra legs, extra arms, extra tentacles, extra wings, extra fins, "
    "missing limbs, missing legs, fused limbs, malformed limbs, mutated limbs, "
    "deformed, disfigured, bad anatomy, cloned body parts, duplicated limbs, "
    "floating limbs, disconnected limbs, extra digits, extra heads, cropped limbs, "
    "text, letters, words, caption, label, title, typography, handwriting, "
    "watermark, signature, logo, page number, printed page, book page"
)

SOLO_NEGATIVE = "multiple creatures, duplicate specimen"

CUTAWAY_NEGATIVE = (
    "cross-section, cross-sectional view, cutaway view, body cut open, sliced open, "
    "split open body, dissection, dissected specimen, exposed viscera, "
    "internal anatomy diagram, x-ray view"
)

GEN_WIDTH, GEN_HEIGHT = 1280, 768
OUT_WIDTH, OUT_HEIGHT = 800, 480

SAMPLER = os.environ.get("COMFYUI_SAMPLER", "euler_ancestral")
SCHEDULER = os.environ.get("COMFYUI_SCHEDULER", "normal")
STEPS = int(os.environ.get("COMFYUI_STEPS", "30"))
CFG = float(os.environ.get("COMFYUI_CFG", "5.0"))


def _build_workflow(prompt, negative_prompt, width, height, seed, steps, cfg):
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
        raise RuntimeError(f"ComfyUI {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from None


_MODEL_SLOTS = (
    ("UNETLoader", "unet_name", DIFFUSION_MODEL),
    ("CLIPLoader", "clip_name", TEXT_ENCODER),
    ("VAELoader", "vae_name", VAE),
)


def check_models():
    rows = []
    for node, field, want in _MODEL_SLOTS:
        try:
            choices = _request(f"/object_info/{node}", timeout=15)[node]["input"]["required"][field][0]
            ok = want in choices
        except Exception:
            ok = False
        rows.append({"node": node, "file": want, "ok": ok})
    return rows


def require_models():
    print("ComfyUI:", COMFYUI_URL)
    missing = []
    for row in check_models():
        print(f"  {row['node']:<12} {row['file']:<32} {'OK' if row['ok'] else 'NOT FOUND'}")
        if not row["ok"]:
            missing.append(row["file"])
    if missing:
        raise RuntimeError(
            "ComfyUI 側にモデルが見つかりません: " + ", ".join(missing)
            + "\nmodels/comfyui/ 以下の配置を CLAUDE.md の Image models 節で確認してください")


def _await_images(prompt_id, timeout):
    deadline = time.time() + timeout
    while True:
        entry = _request(f"/history/{prompt_id}").get(prompt_id)
        status = (entry or {}).get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI generation failed: {json.dumps(status, ensure_ascii=False)[:2000]}")
        images = [img for out in (entry or {}).get("outputs", {}).values()
                  for img in out.get("images", [])]
        if images:
            return images
        if time.time() > deadline:
            raise TimeoutError(f"ComfyUI did not return an image within {timeout}s")
        time.sleep(1)


def get_image(prompt, negative_prompt=NEGATIVE_PROMPT, width=GEN_WIDTH, height=GEN_HEIGHT,
              seed=None, steps=STEPS, cfg=CFG, timeout=600, extra_negative="", solo=True,
              inside_body=False):
    if seed is None:
        seed = random.randint(0, 2 ** 63 - 1)
    negative_prompt = ", ".join(part for part in (
        negative_prompt,
        SOLO_NEGATIVE if solo else "",
        "" if inside_body else CUTAWAY_NEGATIVE,
        extra_negative.strip(),
    ) if part)

    workflow = _build_workflow(f"{STYLE_PREFIX}, {prompt}", negative_prompt, width, height, seed, steps, cfg)
    prompt_id = _request("/prompt", {"prompt": workflow})["prompt_id"]
    images = _await_images(prompt_id, timeout)

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


ITALIC_FONT = os.environ.get("ITALIC_FONT", "NotoSerif-Italic.ttf")

_LATIN_RE = re.compile(r"^[\x20-\x7e\u00a0-\u024f\u1e00-\u1eff]+$")
ITALIC_SHEAR = 0.22


def italicFontFor(text, italic_font):
    if italic_font is not None and _LATIN_RE.match(text or ""):
        return italic_font
    return None


def getItalicWidth(text, upright_font, italic_font):
    font = italicFontFor(text, italic_font)
    if font is not None:
        return getTextWidth(text, font)
    return getTextWidth(text, upright_font) + int(upright_font.getbbox(text)[3] * ITALIC_SHEAR) + 1


def drawItalicText(layer, xy, text, upright_font, italic_font, fill):
    font = italicFontFor(text, italic_font)
    if font is not None:
        ImageDraw.Draw(layer).text(xy, text, font=font, fill=fill, anchor="ls")
        return xy[0] + getTextWidth(text, font)

    box = upright_font.getbbox(text)
    if box[2] <= 0 or box[3] <= 0:
        return xy[0]
    slant = int(box[3] * ITALIC_SHEAR) + 1
    patch = Image.new("RGBA", (box[2] + slant, box[3]), (255, 255, 255, 0))
    ImageDraw.Draw(patch).text((0, 0), text, font=upright_font, fill=fill)
    patch = patch.transform(patch.size, Image.AFFINE,
                            (1, ITALIC_SHEAR, -ITALIC_SHEAR * box[3], 0, 1, 0),
                            resample=Image.BICUBIC)
    layer.alpha_composite(patch, (int(xy[0]), int(xy[1] - upright_font.getmetrics()[0])))
    return xy[0] + box[2] + slant // 2


HEAD_FORBIDDEN = "、。，．・：；？！?!）」』】〉》〕｝ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヵヶ゛゜～"
TAIL_FORBIDDEN = "（「『【〈《〔｛"
WORD_HEADS = ("名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭詞")
WORD_TAILS = ("接尾", "非自立")

_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


def _starts_word(kind, detail, previous):
    previous_kind, previous_detail = previous
    if previous_detail == "括弧開":
        return False
    if detail == "括弧開":
        return True
    if kind not in WORD_HEADS or detail in WORD_TAILS:
        return False
    if kind == "名詞" and previous_kind in ("名詞", "接頭詞"):
        return False
    if kind == "動詞" and previous_detail == "サ変接続":
        return False
    return not (kind == "動詞" and previous_kind == "動詞")


def getWords(text):
    words, previous = [], ("", "")
    for token in tokenizer().tokenize(text):
        kind, detail = (token.part_of_speech.split(",") + [""])[:2]
        if words and not _starts_word(kind, detail, previous):
            words[-1] += token.surface
        else:
            words.append(token.surface)
        previous = (kind, detail)
    return words


def _fitting(word, font, max_width):
    cut = len(word)
    while cut > 1 and getTextWidth(word[:cut], font) > max_width:
        cut -= 1
    while cut < len(word) and word[cut] in HEAD_FORBIDDEN:
        cut += 1
    while cut > 1 and word[cut - 1] in TAIL_FORBIDDEN:
        cut -= 1
    return word[:cut], word[cut:]


def getLineBreak(text, font, max_width):
    lines = [""]
    for word in getWords(text):
        while word:
            if getTextWidth(lines[-1] + word, font) <= max_width:
                lines[-1] += word
                word = ""
            elif lines[-1]:
                lines.append("")
            else:
                lines[-1], word = _fitting(word, font, max_width)
                lines.append("")
    return [line for line in lines if line]


def add_caption(name, description, scientific_name, image, title_font, paragraph_font, scientific_font,
                italic_font=None):
    scientific_prefix, scientific_suffix = " (学名: ", ")"
    description = description.replace("\n", "")

    max_width = 420
    lines = getLineBreak(description, paragraph_font, max_width)

    title_height = getTextHeight(name, title_font) + 7
    line_height = getTextHeight("あ", paragraph_font) + 7
    scientific_height = getTextHeight(f"{scientific_prefix}{scientific_suffix}", scientific_font) + 7
    scientific_width = (getTextWidth(scientific_prefix, scientific_font)
                        + getItalicWidth(scientific_name, scientific_font, italic_font)
                        + getTextWidth(scientific_suffix, scientific_font))
    total_height = line_height * len(lines) + title_height
    max_line_width = max([getTextWidth(l, paragraph_font) for l in lines] + [getTextWidth(name, title_font)+scientific_width])

    im_w, im_h = image.size
    padding = 10

    def place(near_lo, near_hi, far_lo, far_hi, limit):
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
    sci_x = x + getTextWidth(name, title_font)
    sci_y = y + title_height - scientific_height + scientific_font.getmetrics()[0]
    draw.text((sci_x, sci_y), scientific_prefix, font=scientific_font, fill=text_color, anchor="ls")
    sci_x += getTextWidth(scientific_prefix, scientific_font)
    sci_x = drawItalicText(txt_layer, (sci_x, sci_y), scientific_name, scientific_font, italic_font, text_color)
    draw.text((sci_x, sci_y), scientific_suffix, font=scientific_font, fill=text_color, anchor="ls")
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height + title_height), line, font=paragraph_font, fill=text_color)

    return Image.alpha_composite(image.convert('RGBA'), txt_layer).convert("RGB")
