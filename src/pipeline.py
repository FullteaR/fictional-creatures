"""カード1枚を作る手順。ノートブックと Web UI の共通の入口。

産地・種別の表、フォント、名前生成器、カードを1枚作る手順をここに置く。
ノートブック (monster-generator.ipynb) と web/server.py はどちらもここを呼ぶので、
生成の中身を変えるときに直す場所はこのファイルだけでよい。

違いは途中経過の受け取り方だけ:
  emit なし          call_llm が今までどおり stdout にトークンを流す (Web UI 側)
  emit=console_emit  工程名つきで stdout に流す (ノートブック側)
"""

import json
import os
import random
import re
import time
from contextlib import nullcontext
from datetime import datetime

from PIL import ImageFont
from PIL.PngImagePlugin import PngInfo

from MonsterNameGenerator import MarkovMonsterNameGenerator
from imageGenerateUtils import add_caption, get_image
from textGenerateUtils import (draws_group, generate_description, generate_profile,
                               generate_prompt, generate_scientific_name, pick_traits,
                               refine_prompt_with_image, token_sink)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("ENDEMIC_OUT_DIR", os.path.join(SRC_DIR, "endemic"))

FIELDS = [
    "杉林", "古代林", "畑", "草むら", "花畑", "密林", "水没林", "ジャングル", "峠", "山の麓",
    "樹海", "竹林", "森", "霧の森", "熱帯雨林", "サバンナ", "桜並木", "果樹園",
    "洞窟", "鍾乳洞", "谷底", "岩石地帯", "鉱山", "荒野", "岩の中",
    "雪原", "凍土", "氷河",
    "旧市街地", "化学工場跡地", "都市の下水道", "古城", "都市部", "廃工場", "地下鉄廃線", "空中都市",
    "大砂漠", "オアシス",
    "海", "深海", "浅瀬", "砂浜", "汽水域", "川底", "孤島", "海底遺跡", "湖", "潮溜まり",
    "地下水路", "滝", "沈没船", "サンゴ礁",
    "成層圏", "惑星中心部", "溶岩地帯",
    "モンスターの体内",
]
SPECIES = [
    "生物", "鳥", "虫", "植物", "花", "草", "木", "キノコ", "魚", "爬虫類", "哺乳類", "両生類",
    "巨大生物", "小型生物", "草食動物", "肉食動物", "寄生生物", "絶滅危惧種", "甲殻類", "貝",
    "群生生物", "原始生物", "人工生命", "分類不明の生物",
]
# 名前の末尾に種別をくっつけて通りのいい和名にするのは、この4種だけ
SUFFIXABLE_SPECIES = ("貝", "草", "鳥", "魚")

# 工程。UI の進捗表示がこの順番と ID をそのまま使う
STEPS = [
    ("name", "名称"),
    ("description", "解説"),
    ("scientific_name", "学名"),
    ("profile", "裏設定"),
    ("prompt", "作画指示"),
    ("draft", "下書き"),
    ("refine", "指示の練り直し"),
    ("final", "清書"),
    ("caption", "組版"),
]

STEP_LABELS = dict(STEPS)

_INVALID_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


_name_generator = None
_fonts = None


def name_generator():
    """マルコフ連鎖の学習は 4246 行の総当たりで重いので、一度だけやって使い回す"""
    global _name_generator
    if _name_generator is None:
        generator = MarkovMonsterNameGenerator(n=2)
        generator.train_from_file(os.path.join(SRC_DIR, "monsterNames.txt"))
        _name_generator = generator
    return _name_generator


def fonts():
    global _fonts
    if _fonts is None:
        _fonts = {
            "title": ImageFont.truetype(os.path.join(SRC_DIR, "ipagp.ttf"), 27),
            "paragraph": ImageFont.truetype(os.path.join(SRC_DIR, "ipagp.ttf"), 15),
            "caption": ImageFont.truetype(os.path.join(SRC_DIR, "ipagp.ttf"), 12),
            "italic": ImageFont.truetype(os.path.join(SRC_DIR, "NotoSerif-Italic.ttf"), 12),
        }
    return _fonts


def build_target(field=None, species=None, name=None):
    """産地・種別・名前から「〜にて観測される架空の〜「〜」」を組み立てる"""
    generator = name_generator()
    given = (name or "").strip()
    name = given or generator.generate()
    field = field or random.choice(FIELDS)
    species = species or random.choice(SPECIES)
    if field == "モンスターの体内":
        field = generator.generate() + "の体内"
    # 名前を指定して呼ばれたときは、指定どおりの名前で出す
    if not given and species in SUFFIXABLE_SPECIES and random.randint(0, 1) == 1:
        name = name + species
    return name, field, species, "{0}にて観測される架空の{1}「{2}」".format(field, species, name)


def traits_payload(traits):
    """裏設定のうち UI に見せる分。directive などの英文はカード詳細で出す"""
    return {
        "danger": traits["danger"],
        "population": traits["population"]["label"],
        "group": bool(traits["population"]["group"]),
        "composition": traits["composition"]["label"],
        "directive": traits["composition"]["directive"],
    }


def _png_metadata(card):
    """カードの情報を PNG のテキストチャンクに入れる。

    PNG に EXIF はまず使われず、この界隈 (ComfyUI / A1111) の通り相場は
    tEXt / iTXt チャンク。iTXt は UTF-8 なので和文がそのまま入る
    (tEXt は Latin-1 なので入らない)。Title / Description / Creation Time /
    Software は PNG 仕様の標準キーワードで、汎用のビューアでも読める。
    Endemic だけはこちらの都合なので、まとめて JSON で持たせる。
    """
    info = PngInfo()
    info.add_itxt("Title", card["name"])
    info.add_itxt("Description", card["description"])
    info.add_itxt("Creation Time", card["created_at"])
    info.add_itxt("Software", "fictional-creatures")
    info.add_itxt("Endemic", json.dumps(card, ensure_ascii=False))
    return info


def generate_card(emit=None, *, name=None, field=None, species=None, out_dir=OUT_DIR):
    """カードを1枚作って保存し、(メタデータの dict, 仕上がりの画像) を返す。

    emit(event)  進捗イベントを受け取る関数。省略すると call_llm が
                 今までどおり stdout にトークンを流す
    """
    started = time.time()
    notify = emit if emit is not None else (lambda event: None)

    def sink(key):
        # emit が無いときは差し替えない。call_llm の既定どおり stdout に出る
        if emit is None:
            return nullcontext()
        return token_sink(lambda text: emit({"type": "token", "step": key, "text": text}))

    def run(key, work):
        notify({"type": "step", "step": key, "status": "running"})
        with sink(key):
            value = work()
        notify({"type": "step", "step": key, "status": "done"})
        return value

    def field_done(key, value):
        notify({"type": "field", "name": key, "value": value})

    name, field, species, target = build_target(field, species, name)
    traits = pick_traits(species)
    notify({"type": "start", "name": name, "field": field, "species": species,
            "target": target, "traits": traits_payload(traits)})
    notify({"type": "step", "step": "name", "status": "done"})
    field_done("name", name)

    description = run("description", lambda: generate_description(target))
    description = description.strip()
    field_done("description", description)

    scientific_name = run("scientific_name",
                          lambda: generate_scientific_name(target, description)).strip()
    field_done("scientific_name", scientific_name)

    # 説明文の後に作る。説明文と矛盾しない範囲で、説明文に無い見た目の細部を足す役
    profile = run("profile", lambda: generate_profile(target, description, traits)).strip()
    field_done("profile", profile)

    prompt = run("prompt", lambda: generate_prompt(target, description, profile, traits)).strip()
    field_done("prompt", prompt)

    extra_negative = traits["composition"]["negative"]
    solo = not draws_group(traits)  # 群れる生物は複数個体を落とさない

    draft = run("draft", lambda: get_image(prompt, extra_negative=extra_negative, solo=solo))
    refined = run("refine", lambda: refine_prompt_with_image(
        prompt, draft, target, description, profile, traits)).strip()
    field_done("refined_prompt", refined)

    background = run("final", lambda: get_image(refined, extra_negative=extra_negative, solo=solo))

    final_image = run("caption", lambda: add_caption(
        name, description, scientific_name, background,
        fonts()["title"], fonts()["paragraph"], fonts()["caption"], fonts()["italic"]))

    os.makedirs(out_dir, exist_ok=True)
    stem = "{0}-{1}".format(datetime.now().strftime("%Y%m%d-%H%M%S"),
                            _INVALID_FILENAME.sub("_", name) or "無名")
    card = {
        "image": stem + ".png",
        "name": name,
        "scientific_name": scientific_name,
        "description": description,
        "field": field,
        "species": species,
        "target": target,
        "traits": traits_payload(traits),
        "profile": profile,
        "prompt": prompt,
        "refined_prompt": refined,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
    }
    final_image.save(os.path.join(out_dir, stem + ".png"), pnginfo=_png_metadata(card))

    notify({"type": "done", "card": card})
    return card, final_image


def console_emit(event):
    """ノートブック向けの emit。工程名を挟みながら stdout に流す"""
    kind = event["type"]
    if kind == "start":
        traits = event["traits"]
        print(event["target"])
        print("{0} / {1} / {2}".format(traits["composition"], traits["danger"], traits["population"]))
    elif kind == "step" and event["status"] == "running":
        print("\n[{0}]".format(STEP_LABELS.get(event["step"], event["step"])))
    elif kind == "token":
        print(event["text"], end="", flush=True)
    elif kind == "done":
        print("-> {0}".format(event["card"]["image"]))
