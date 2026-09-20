"""カード1枚を作る手順。ノートブックと Web UI の共通の入口。"""

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
from textGenerateUtils import (apply_proof, apply_review, draws_solo, extra_negative,
                               generate_description, generate_profile, generate_prompt,
                               generate_scientific_name, pick_traits, review_description,
                               review_image, review_payload, token_sink)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("ENDEMIC_OUT_DIR", os.path.join(SRC_DIR, "endemic"))
REVIEW_RETRIES = int(os.environ.get("ENDEMIC_REVIEW_RETRIES", "1"))

FIELDS = [
    "杉林", "古代林", "畑", "草むら", "花畑", "密林", "水没林", "ジャングル", "峠", "山の麓",
    "樹海", "竹林", "森", "霧の森", "熱帯雨林", "サバンナ", "桜並木", "果樹園",
    "笹薮", "ブナ林", "マングローブ林",
    "湿原", "泥炭地", "ヨシ原", "水田", "棚田", "牧草地", "高原の草原",
    "洞窟", "鍾乳洞", "谷底", "岩石地帯", "鉱山", "荒野", "岩の中",
    "高山帯", "断崖の岩棚", "尾根", "風穴", "雲海の上",
    "火口", "溶岩洞", "地熱地帯", "断層の割れ目", "隕石孔", "間欠泉",
    "雪原", "凍土", "氷河", "雪渓", "流氷", "氷床の下", "永久凍土の割れ目",
    "旧市街地", "化学工場跡地", "都市の下水道", "古城", "都市部", "廃工場", "地下鉄廃線", "空中都市",
    "図書館の書庫", "倉庫の奥", "配管の中", "送電鉄塔", "廃校", "地下駐車場", "ごみ集積場",
    "研究所跡", "墓地", "貯水槽", "温室",
    "大砂漠", "オアシス", "塩湖", "塩の平原", "砂丘", "涸れ川",
    "海", "深海", "浅瀬", "砂浜", "汽水域", "川底", "孤島", "海底遺跡", "湖", "潮溜まり",
    "地下水路", "滝", "沈没船", "サンゴ礁",
    "上空の雷雲", "積乱雲の中", "電離層", "季節風の通り道",
    "成層圏", "惑星中心部", "溶岩地帯",
    "落ち葉の下", "樹皮の裏", "朽ちた切り株", "岩の割れ目", "苔むした倒木", "巨木のうろ",
    "獣の毛の中", "キノコの傘の裏",
    "モンスターの体内",
]
SPECIES = [
    "生物", "鳥", "虫", "植物", "花", "草", "木", "キノコ", "魚", "爬虫類", "哺乳類", "両生類",
    "巨大生物", "小型生物", "草食動物", "肉食動物", "寄生生物", "絶滅危惧種", "甲殻類", "貝",
    "群生生物", "原始生物", "人工生命", "分類不明の生物",
    "軟体動物", "刺胞動物", "棘皮動物", "環形動物", "菌類", "粘菌", "藻類", "苔", "地衣類", "微生物",
    "夜行性生物", "穴居生物", "滑空生物", "濾過摂食生物", "腐食性生物", "共生生物", "擬態生物",
    "回遊性の生物", "変温生物",
    "外来種", "家畜化された生物", "半水生生物", "樹上生物", "地中生物",
]
SUFFIXABLE_SPECIES = ("貝", "草", "鳥", "魚", "虫", "苔")

STEPS = [
    ("name", "名称"),
    ("description", "解説"),
    ("scientific_name", "学名"),
    ("profile", "裏設定"),
    ("prompt", "作画指示"),
    ("draft", "下書き"),
    ("review", "照合"),
    ("final", "清書"),
    ("proof", "校正"),
    ("caption", "組版"),
]

STEP_LABELS = dict(STEPS)

_INVALID_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


_name_generator = None
_fonts = None


def name_generator():
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
    generator = name_generator()
    given = (name or "").strip()
    name = given or generator.generate()
    field = field or random.choice(FIELDS)
    species = species or random.choice(SPECIES)
    if field == "モンスターの体内":
        field = generator.generate() + "の体内"
    if not given and species in SUFFIXABLE_SPECIES and random.randint(0, 1) == 1:
        name = name + species
    return name, field, species, "{0}にて観測される架空の{1}「{2}」".format(field, species, name)


def _merge_negative(notes, found):
    parts = (part.strip() for part in ", ".join((notes, found)).split(","))
    return ", ".join(dict.fromkeys(part for part in parts if part))


def _unmatched(entry):
    return sum(1 for item in entry["review"] if not item["ok"])


def traits_payload(traits):
    return {
        "danger": traits["danger"],
        "population": traits["population"]["label"],
        "group": bool(traits["population"]["group"]),
        "body_plan": traits["body_plan"]["label"],
        "register": traits["register"]["label"],
        "composition": traits["composition"]["label"],
        "directive": traits["composition"]["directive"],
    }


def _png_metadata(card):
    info = PngInfo()
    info.add_itxt("Title", card["name"])
    info.add_itxt("Description", card["description"])
    info.add_itxt("Creation Time", card["created_at"])
    info.add_itxt("Software", "fictional-creatures")
    info.add_itxt("Endemic", json.dumps(card, ensure_ascii=False))
    return info


def generate_card(emit=None, *, name=None, field=None, species=None, out_dir=OUT_DIR):
    started = time.time()
    notify = emit if emit is not None else (lambda event: None)

    def sink(key):
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

    description = run("description", lambda: generate_description(target, traits))
    description = description.strip()
    field_done("description", description)

    scientific_name = run("scientific_name",
                          lambda: generate_scientific_name(target, description)).strip()
    field_done("scientific_name", scientific_name)

    profile = run("profile", lambda: generate_profile(target, description, traits)).strip()
    field_done("profile", profile)

    prompt = run("prompt", lambda: generate_prompt(target, description, profile, traits)).strip()
    field_done("prompt", prompt)

    negative = extra_negative(traits)
    solo = draws_solo(traits)

    seed = random.getrandbits(63)
    current, notes, plates, rounds = prompt, "", [], []
    while True:
        plate_negative = ", ".join(part for part in (negative, notes) if part)
        plates.append(run("draft" if not rounds else "final", lambda: get_image(
            current, extra_negative=plate_negative, solo=solo, seed=seed)))
        review = run("review", lambda: review_image(
            plates[-1], current, target, description, profile, traits))
        rounds.append({"prompt": current, "negative": notes, "review": review_payload(review)})
        if all(entry["ok"] for entry in review) or len(rounds) > REVIEW_RETRIES:
            break
        if len(rounds) > 1 and _unmatched(rounds[-1]) >= _unmatched(rounds[-2]):
            break
        fixed, found = apply_review(current, review)
        merged = _merge_negative(notes, found)
        if fixed == current and merged == notes:
            break
        current, notes = fixed, merged

    chosen = min(range(len(rounds)), key=lambda index: _unmatched(rounds[index]))
    background = plates[chosen]
    refined = rounds[chosen]["prompt"]
    field_done("refined_prompt", refined)

    proof = run("proof", lambda: review_description(
        background, description, target, species, field, traits))
    draft_description, description = description, apply_proof(description, proof)
    if description != draft_description:
        field_done("description", description)

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
        "draft_description": draft_description,
        "field": field,
        "species": species,
        "target": target,
        "traits": traits_payload(traits),
        "profile": profile,
        "prompt": prompt,
        "refined_prompt": refined,
        "rounds": rounds,
        "chosen_round": chosen,
        "seed": seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
    }
    final_image.save(os.path.join(out_dir, stem + ".png"), pnginfo=_png_metadata(card))

    notify({"type": "done", "card": card})
    return card, final_image


def console_emit(event):
    kind = event["type"]
    if kind == "start":
        traits = event["traits"]
        print(event["target"])
        print("{0} / {1} / {2}".format(
            traits["body_plan"], traits["register"], traits["composition"]))
        print("{0} / {1}".format(traits["danger"], traits["population"]))
    elif kind == "step" and event["status"] == "running":
        print("\n[{0}]".format(STEP_LABELS.get(event["step"], event["step"])))
    elif kind == "token":
        print(event["text"], end="", flush=True)
    elif kind == "done":
        print("-> {0}".format(event["card"]["image"]))
