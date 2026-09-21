import os
import base64
import contextvars
import random
import re

import Levenshtein
from contextlib import contextmanager
from io import BytesIO
from openai import OpenAI
from sampleMonsters import *

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080/v1")
_client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="dummy")
_MODEL = "local-model"


_token_sink = contextvars.ContextVar("token_sink", default=None)


@contextmanager
def token_sink(sink):
    handle = _token_sink.set(sink)
    try:
        yield
    finally:
        _token_sink.reset(handle)


def _emit(text):
    sink = _token_sink.get()
    if sink is None:
        print(text, end="", flush=True)
    else:
        sink(text)


def call_llm(messages, max_tokens=2048):
    stream = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        stream=True
    )
    chunks = []
    finish_reason = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta.content or ""
        if delta:
            _emit(delta)
            chunks.append(delta)
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    _emit("\n")
    if finish_reason == "length":
        _emit(f"[warn] max_tokens={max_tokens} に到達して打ち切られました（繰り返しループの可能性）\n")
    return "".join(chunks)


def first_line(text):
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


_NUMBER = r"(?:\d+|no|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_LIMB_PART = (r"(?:(?:fore|hind|mid|middle|walking|swimming)[- ]?)?"
              r"(?:legs?|limbs?|arms?|tentacles?|wings?|fins?|claws?|pincers?|antennae|chelipeds?|appendages?)")
_LIMB_COUNT_RE = re.compile(
    rf"\b{_NUMBER}(?:\s+pairs?\s+of)?(?:\s+[a-z-]+){{0,3}}\s+{_LIMB_PART}(?:\s+on each side)?\b",
    re.IGNORECASE,
)


def limb_count_phrases(prompt):
    seen = []
    for match in _LIMB_COUNT_RE.findall(prompt):
        phrase = " ".join(match.split())
        if phrase.lower() not in [s.lower() for s in seen]:
            seen.append(phrase)
    return seen


BODY_PLANS = [
    {"weight": 14, "label": "四肢型", "kind": "脊椎",
     "ja": "頭と胴と尾があり、4本の脚で体を支える背骨のある体",
     "en": "four-limbed vertebrate body, head and neck and tail, weight carried on four legs",
     "limbs": "count", "head": True, "limb_cap": 4, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 12, "label": "節足型", "kind": "節足",
     "ja": "かたい外骨格と関節のある脚を持ち、体がいくつかの節に分かれている",
     "en": "segmented arthropod body, hard jointed exoskeleton, legs arranged along both sides",
     "limbs": "count", "head": True, "limb_cap": 8, "limb_heavy": True, "colonial": False,
     "negative": ""},
    {"weight": 8, "label": "翼のある四肢型", "kind": "翼",
     "ja": "2本の翼と2本の脚を持ち、体が羽毛か皮膜に覆われている",
     "en": "winged vertebrate body, two wings and two legs, covered in feathers or membrane",
     "limbs": "count", "head": True, "limb_cap": 4, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 8, "label": "紡錘型の遊泳体", "kind": "遊泳",
     "ja": "脚を持たず、ひれと尾で水中を泳ぐ流線型の体",
     "en": "streamlined swimming body, paired fins and a tail fin, no legs",
     "limbs": "count", "head": True, "limb_cap": 6, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 8, "label": "軟体型", "kind": "軟体",
     "ja": "骨格を持たず、やわらかい胴から触手を伸ばす体",
     "en": "soft boneless body, rounded mantle, tentacles trailing beneath it",
     "limbs": "count", "head": True, "limb_cap": 6, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 7, "label": "長い無脚型", "kind": "蛇",
     "ja": "脚が無く、細長い胴をくねらせて進む体",
     "en": "long limbless serpentine body, one smooth continuous trunk, no legs and no arms",
     "limbs": "none", "head": True, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 7, "label": "植物体型", "kind": "植物",
     "ja": "茎と葉を広げ、地面に根を張って立つ体",
     "en": "rooted plant body, roots and stem and leaves, standing anchored in the ground",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": "walking, animal face, paws, claws"},
    {"weight": 6, "label": "殻を持つ型", "kind": "殻",
     "ja": "らせん状あるいは二枚の殻に体を収め、やわらかい足だけを外に出している",
     "en": "shelled body, hard coiled or paired shell, one soft muscular foot protruding, "
           "no legs and no arms",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 6, "label": "放射相称型", "kind": "放射",
     "ja": "前後左右の別が無く、中心から等しく腕が広がる体",
     "en": "radially symmetric body, no head and no front or back, arms spreading evenly "
           "from a central disc",
     "limbs": "count", "head": False, "limb_cap": 8, "limb_heavy": True, "colonial": False,
     "negative": "head, face, snout, bilateral symmetry"},
    {"weight": 6, "label": "袋状・球状型", "kind": "原始",
     "ja": "頭も脚も無い、単純な袋あるいは球のかたちの体",
     "en": "simple sac-like or spherical body, no head, no limbs, plain closed surface",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": "head, face, snout, eyes, mouth, tail"},
    {"weight": 6, "label": "固着型", "kind": "固着",
     "ja": "柄や付着器で一か所に体を固定し、動かずに暮らす",
     "en": "sessile body anchored to the substrate by a stalk or holdfast, fixed in place, "
           "no legs and no arms",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": ""},
    {"weight": 5, "label": "菌類型", "kind": "菌",
     "ja": "菌糸を広げ、傘と柄のような子実体をつくる体",
     "en": "fungal body, capped fruiting bodies on short stalks rising from spreading mycelium",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": "walking, animal face, paws, claws"},
    {"weight": 5, "label": "群体型", "kind": "群体",
     "ja": "同じ小さな単位が多数集まって群体をつくり、全体としてひとつの輪郭を持たない",
     "en": "colony of many identical small units packed side by side, an irregular spreading "
           "mass with no single body outline and no head anywhere on it",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": True,
     "negative": "single large animal, one large creature, animal body, animal face, snout, "
                 "head, tail, torso, quadruped, four legs, mammal, reptile"},
    {"weight": 5, "label": "膜状・帯状型", "kind": "軟体",
     "ja": "薄い膜あるいは帯のように平たく、ひらひらと動く体",
     "en": "flat sheet-like body, thin rippling membrane, no distinct limbs",
     "limbs": "none", "head": False, "limb_cap": 0, "limb_heavy": False, "colonial": False,
     "negative": "animal face, snout, jointed legs"},
]
DEFAULT_KINDS = tuple(dict.fromkeys(row["kind"] for row in BODY_PLANS))
SPECIES_KINDS = {
    "鳥": ("翼",),
    "虫": ("節足", "翼"),
    "甲殻類": ("節足",),
    "魚": ("遊泳",),
    "貝": ("殻", "軟体"),
    "爬虫類": ("脊椎", "蛇"),
    "哺乳類": ("脊椎",),
    "両生類": ("脊椎",),
    "植物": ("植物", "固着"),
    "花": ("植物",),
    "草": ("植物",),
    "木": ("植物",),
    "キノコ": ("菌",),
    "草食動物": ("脊椎", "翼"),
    "肉食動物": ("脊椎", "翼", "節足"),
    "群生生物": ("群体", "固着"),
    "原始生物": ("原始", "軟体", "群体"),
    "軟体動物": ("軟体", "殻"),
    "刺胞動物": ("放射", "軟体"),
    "棘皮動物": ("放射",),
    "環形動物": ("蛇", "原始"),
    "菌類": ("菌", "群体"),
    "粘菌": ("群体", "原始"),
    "藻類": ("植物", "群体"),
    "苔": ("植物", "群体"),
    "地衣類": ("群体", "固着"),
    "微生物": ("原始", "群体"),
    "滑空生物": ("翼", "脊椎"),
    "濾過摂食生物": ("固着", "殻", "軟体", "放射"),
    "回遊性の生物": ("遊泳", "翼", "軟体"),
    "半水生生物": ("脊椎", "遊泳", "蛇"),
    "樹上生物": ("脊椎", "翼", "節足"),
}
_unknown_kinds = {k for kinds in SPECIES_KINDS.values() for k in kinds} - set(DEFAULT_KINDS)
assert not _unknown_kinds, f"SPECIES_KINDS に BODY_PLANS に無い kind があります: {_unknown_kinds}"

COMPOSITIONS = [
    {"weight": 50, "label": "生物の全体図",
     "directive": "the whole creature centred in frame, entire body visible end to end, "
                  "side-on specimen view, habitat kept plain and secondary",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": True,
     "shows_group": True, "group_only": False, "solo": True, "palette": "",
     "negative": "cropped, out of frame, extreme close-up"},
    {"weight": 20, "label": "生息地の風景",
     "directive": "wide view of the habitat filling the frame, the creature small and partly "
                  "concealed within the scene, environment shown in full",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": False,
     "shows_group": True, "group_only": False, "solo": True, "palette": "",
     "negative": "extreme close-up, empty scenery"},
    {"weight": 15, "label": "生態の痕跡",
     "directive": "the creature itself absent from frame, no animal visible, only {trace} "
                  "left behind, shown in situ in the empty habitat",
     "draws_creature": False, "limb_mode": "absent", "magnifies_body": False,
     "shows_group": False, "group_only": False, "solo": True, "palette": "",
     "negative": "live animal, living creature, animal, eyes, face, moving limbs"},
    {"weight": 15, "label": "体の一部の拡大図",
     "directive": "close-up study of {part} filling the frame, the rest of the body out of "
                  "frame, habitat plain and out of focus behind",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": True,
     "shows_group": False, "group_only": False, "solo": True, "palette": "",
     "negative": "full body, whole creature, wide shot, distant view"},
    {"weight": 50, "label": "群れの遠景",
     "directive": "distant wide view of a dense swarm of the species massed across the habitat, "
                  "many small individuals scattered and clustered far from the viewer, each one "
                  "tiny and without visible detail, the habitat visible around and beyond them",
     "draws_creature": True, "limb_mode": "distant", "magnifies_body": False,
     "shows_group": True, "group_only": True, "solo": False, "palette": "",
     "negative": "close-up, macro, single specimen, large creature in foreground, portrait, "
                 "human, people, person, crowd, humanoid figure, standing figures, "
                 "buildings, vehicles"},
    {"weight": 10, "label": "標本図",
     "directive": "the organism preserved as a dried museum specimen laid flat on a plain "
                  "neutral board, the whole body spread out so every part is visible, "
                  "no habitat and no ground",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": True,
     "shows_group": False, "group_only": False, "solo": True, "palette": "",
     "negative": "habitat, foliage, sky, water, scenery, motion blur, running, flying"},
    {"weight": 10, "label": "擬態と保護色",
     "directive": "the creature concealed against its surroundings by camouflage, its outline "
                  "broken up and matching the colour and texture of the habitat, only part of "
                  "the body separable from the background, the rest of the scene ordinary",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": False,
     "shows_group": False, "group_only": False, "solo": True, "palette": "drab",
     "negative": "creature isolated on plain background, centred portrait, "
                 "high contrast subject, spotlight on the animal"},
    {"weight": 8, "label": "幼体と成体の比較",
     "directive": "two individuals of the same species side by side on a plain background, "
                  "a small juvenile on one side and the full-grown adult on the other, both "
                  "whole and in the same side-on pose, the difference in size and proportion clear",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": True,
     "shows_group": False, "group_only": False, "solo": False, "palette": "",
     "negative": "wide shot, habitat scene, crowd, many individuals, different species, "
                 "family scene"},
    {"weight": 8, "label": "夜間の観察",
     "directive": "the creature at night, the habitat around it dark and flat, cool pale "
                  "night palette, the animal the lightest shape in the frame, everything "
                  "rendered evenly and without glare",
     "draws_creature": True, "limb_mode": "count", "magnifies_body": False,
     "shows_group": True, "group_only": False, "solo": True, "palette": "light",
     "negative": "daylight, blue sky, sunlight, glowing, bioluminescence, light beams, "
                 "spotlight, harsh shadows, crushed black"},
]
TRACES = [
    "the picked-over remains of its prey with feeding marks",
    "an abandoned nest of gathered debris",
    "a split and empty shed exoskeleton",
    "a trail of footprints pressed into soft ground",
    "a cluster of eggs attached to the substrate",
    "gnawed plant stems and scattered fragments",
    "burrow openings worn into the substrate",
]
PARTS = [
    {"text": "the head", "needs": "head"},
    {"text": "the mouthparts", "needs": "head"},
    {"text": "one limb", "needs": "limbs"},
    {"text": "the patterned surface of the body", "needs": ""},
    {"text": "the sensory organs", "needs": ""},
]
LIMB_HEAVY_SPECIES = ("甲殻類",)
LIMB_HEAVY_WEIGHT = 0.0
MAX_LIMBS = 8

SIZES = [
    {"weight": 8, "label": "微小", "ja": "数ミリ以下",
     "en": "only a few millimetres long, smaller than the debris around it"},
    {"weight": 20, "label": "数センチ", "ja": "数センチ程度",
     "en": "a few centimetres long"},
    {"weight": 25, "label": "数十センチ", "ja": "数十センチ程度",
     "en": "a few tens of centimetres long"},
    {"weight": 22, "label": "1メートル", "ja": "1メートル前後",
     "en": "about a metre long"},
    {"weight": 15, "label": "数メートル", "ja": "数メートル",
     "en": "several metres long, larger than everything around it"},
    {"weight": 10, "label": "巨大", "ja": "十メートルを超える",
     "en": "over ten metres long, dwarfing its surroundings"},
]
SPECIES_SIZES = {
    "微生物": ("微小",),
    "小型生物": ("微小", "数センチ"),
    "虫": ("微小", "数センチ"),
    "巨大生物": ("数メートル", "巨大"),
}
_unsized = {label for labels in SPECIES_SIZES.values() for label in labels} - {
    row["label"] for row in SIZES}
assert not _unsized, f"SPECIES_SIZES に SIZES に無い label があります: {_unsized}"

PALETTES = [
    {"weight": 14, "label": "土と枯葉", "kinds": (),
     "ja": "土や枯れ葉に似た茶褐色を基調とし",
     "en": "muted earth brown and dead-leaf ochre palette",
     "conspicuous": False, "dark": False},
    {"weight": 12, "label": "苔と葉", "kinds": (),
     "ja": "苔や葉に似た緑を基調とし",
     "en": "muted moss green and olive palette",
     "conspicuous": False, "dark": False},
    {"weight": 12, "label": "岩と灰", "kinds": (),
     "ja": "岩や灰に似た灰色を基調とし",
     "en": "muted stone grey and ash palette",
     "conspicuous": False, "dark": False},
    {"weight": 10, "label": "骨と雪", "kinds": (),
     "ja": "骨や雪のような白を基調とし",
     "en": "pale bone white and chalk palette",
     "conspicuous": False, "dark": False},
    {"weight": 8, "label": "墨", "kinds": (),
     "ja": "墨のような黒を基調とし",
     "en": "near-black charcoal and soot palette",
     "conspicuous": False, "dark": True},
    {"weight": 8, "label": "深い青", "kinds": (),
     "ja": "藍や紺のような深い青を基調とし",
     "en": "deep muted indigo and slate blue palette",
     "conspicuous": False, "dark": False},
    {"weight": 8, "label": "錆と赤土", "kinds": (),
     "ja": "錆や赤土のような赤褐色を基調とし",
     "en": "muted rust red and oxidised orange palette",
     "conspicuous": True, "dark": False},
    {"weight": 7, "label": "黄土と琥珀", "kinds": (),
     "ja": "黄土や琥珀のような黄色を基調とし",
     "en": "dull ochre yellow and amber palette",
     "conspicuous": True, "dark": False},
    {"weight": 7, "label": "半透明", "kinds": ("軟体", "原始", "群体", "遊泳", "殻", "放射", "菌", "節足"),
     "ja": "色素がほとんど無く半透明で",
     "en": "translucent and almost colourless, only faintly tinted",
     "conspicuous": False, "dark": False},
    {"weight": 6, "label": "紫紺", "kinds": (),
     "ja": "紫紺やすみれ色のような紫を基調とし",
     "en": "muted violet and dusty purple palette",
     "conspicuous": True, "dark": False},
    {"weight": 5, "label": "警告色", "kinds": (),
     "ja": "全体は地味な体色で、一か所だけに鮮やかな警告色の斑があり",
     "en": "drab grey-brown body kept unsaturated except for one small red warning patch",
     "conspicuous": True, "dark": False},
    {"weight": 5, "label": "白黒の斑", "kinds": (),
     "ja": "白と黒のはっきりした斑や縞に分かれ",
     "en": "stark black and white patterned markings",
     "conspicuous": True, "dark": False},
]
SURFACES = [
    {"weight": 14, "label": "なめらか", "kinds": (),
     "ja": "つやの無いなめらかな体表をしている",
     "en": "smooth matte surface without gloss"},
    {"weight": 10, "label": "細かな毛", "kinds": ("脊椎", "翼", "節足", "植物", "菌"),
     "ja": "全身が細かい毛に覆われている",
     "en": "covered all over in fine short hair"},
    {"weight": 10, "label": "鱗", "kinds": ("脊椎", "蛇", "遊泳"),
     "ja": "重なり合う鱗に覆われている",
     "en": "overlapping scales covering the body"},
    {"weight": 10, "label": "硬い殻質", "kinds": ("節足", "殻", "放射", "固着"),
     "ja": "硬い殻質の板に覆われている",
     "en": "hard chitinous plates over the body"},
    {"weight": 9, "label": "粘液", "kinds": ("軟体", "原始", "群体", "蛇", "菌", "遊泳"),
     "ja": "薄い粘液に覆われてぬめる",
     "en": "damp mucous film over the surface, matte and not shiny"},
    {"weight": 9, "label": "粉を吹く", "kinds": ("植物", "菌", "節足", "群体", "固着"),
     "ja": "乾いてざらつき、白い粉を吹いている",
     "en": "dry powdery bloom dusting the surface"},
    {"weight": 9, "label": "棘と突起", "kinds": (),
     "ja": "細かな棘や突起が一面に生えている",
     "en": "covered in fine spines and blunt tubercles"},
    {"weight": 8, "label": "革質のしわ", "kinds": ("脊椎", "蛇", "軟体", "原始", "翼"),
     "ja": "しわの寄った厚い革質の体表をしている",
     "en": "thick wrinkled leathery hide"},
    {"weight": 7, "label": "浮き出た脈", "kinds": ("軟体", "植物", "菌", "群体", "原始"),
     "ja": "体表に網目状の脈が浮き出ている",
     "en": "raised network of veins across the surface"},
    {"weight": 7, "label": "ひび割れ", "kinds": ("植物", "固着", "殻", "群体", "菌"),
     "ja": "地衣のようにひび割れた硬い表面をしている",
     "en": "cracked crust mottled like lichen"},
    {"weight": 7, "label": "繊毛", "kinds": ("原始", "群体", "軟体", "放射", "固着"),
     "ja": "細かな繊毛が一面に生えている",
     "en": "dense fringe of fine cilia over the whole surface"},
    {"weight": 6, "label": "付着物", "kinds": (),
     "ja": "砂粒や結晶が体表に付着している",
     "en": "grains of sand and small crystals stuck to the surface"},
]
_unknown_look_kinds = ({kind for row in PALETTES + SURFACES for kind in row["kinds"]}
                       - set(DEFAULT_KINDS))
assert not _unknown_look_kinds, \
    f"PALETTES/SURFACES に BODY_PLANS に無い kind があります: {_unknown_look_kinds}"
_lookless = {kind for kind in DEFAULT_KINDS for rows in (PALETTES, SURFACES)
             if not [row for row in rows if not row["kinds"] or kind in row["kinds"]]}
assert not _lookless, f"PALETTES/SURFACES に候補の無い kind があります: {_lookless}"

DANGERS = [
    {"weight": 25, "label": "人間には全く無害"},
    {"weight": 35, "label": "刺激すると刺す、あるいは咬む程度"},
    {"weight": 25, "label": "毒を持ち、接触すると危険"},
    {"weight": 15, "label": "致死的で、接近そのものが極めて危険"},
]
POPULATIONS = [
    {"weight": 20, "label": "大量発生しており、生息地では群れに出くわす", "group": True},
    {"weight": 30, "label": "生息地では普通に見られる", "group": False},
    {"weight": 30, "label": "限られた場所にのみ局所的に生息する", "group": False},
    {"weight": 20, "label": "記録が数例しかない希少種", "group": False},
]
GROUP_DIRECTIVE = ("many individuals of the same species together in the scene, "
                   "one specimen in the foreground shown clearly and in full, "
                   "the others smaller and further back, all identical in form")

_OPENING = ("一文目はその生物が何であるかを名詞で短く言い切ってください"
            "（例:「洞窟の壁に張り付いて生活している生物。」「水に擬態した原生生物。」）。"
            "図鑑の解説文なので、全体を三人称で書いてください。採集や観察の経緯に触れるときは"
            "「採集されている」「報告がある」のように、誰の行為とも特定しない書き方にしてください。"
            "日付や体の細部は二文目以降に回してください。")
_BURIRIA = "古代の湖にて観測される甲殻類「ブリリア」"
_MIZU = "深海にて観測される架空の生物「ミズモドキ」"
_ZATON = "アンカラ洞窟にて観測される架空の生物「ザトン」"
_PLAN_LABELS = {row["label"]: row for row in BODY_PLANS}
_SAMPLE_PLANS = {_BURIRIA: "殻を持つ型", _MIZU: "袋状・球状型", _ZATON: "節足型"}
_unknown_labels = set(_SAMPLE_PLANS.values()) - set(_PLAN_LABELS)
assert not _unknown_labels, f"_SAMPLE_PLANS に BODY_PLANS に無い label があります: {_unknown_labels}"
REGISTERS = [
    {"weight": 50, "label": "図鑑の記述",
     "instruction": "観察された事実だけを淡々と、図鑑の解説文の調子で書いてください。",
     "samples": ((_BURIRIA, EsukaKnight), (_MIZU, Mizumodoki), (_ZATON, Kyomuton))},
    {"weight": 10, "label": "土地の伝承",
     "instruction": "その土地に伝わる言い伝えや俗信を交えて、"
                    "どう呼ばれ、どう扱われてきたかを書いてください。",
     "samples": ((_ZATON, KyomutonDenshou), (_MIZU, MizumodokiDenshou))},
    {"weight": 25, "label": "研究の記録",
     "instruction": "記載や発見の経緯、計測値、まだ分かっていない点を交えて書いてください。",
     "samples": ((_ZATON, KyomutonKiroku), (_MIZU, MizumodokiKiroku))},
]
_unkeyed_samples = ({phrase for row in REGISTERS for phrase, _ in row["samples"]}
                    - set(_SAMPLE_PLANS))
assert not _unkeyed_samples, f"_SAMPLE_PLANS に体型の無い例文があります: {_unkeyed_samples}"


def _parts(plan):
    return [row["text"] for row in PARTS
            if not row["needs"]
            or (row["needs"] == "head" and plan["head"])
            or (row["needs"] == "limbs" and plan["limbs"] != "none")]


def _composition(row, plan):
    picked = dict(row)
    picked["directive"] = row["directive"].format(
        trace=random.choice(TRACES), part=random.choice(_parts(plan)))
    return picked


def _roll(rows, fits=None, weight=lambda row: row["weight"]):
    pool = rows
    if fits:
        pool = [row for row in rows if fits(row)] or rows
    weights = [weight(row) for row in pool]
    if sum(weights) <= 0:
        return random.choice(pool)
    return random.choices(pool, weights=weights)[0]


def _for_kind(rows, kind):
    return [row for row in rows if not row["kinds"] or kind in row["kinds"]]


def _composition_weight(row, species, body_plan):
    if row["magnifies_body"] and (body_plan["limb_heavy"] or species in LIMB_HEAVY_SPECIES):
        return row["weight"] * LIMB_HEAVY_WEIGHT
    return row["weight"]


def _palette_fits(row, composition):
    wanted = composition["palette"]
    return not ((wanted == "drab" and row["conspicuous"])
                or (wanted == "light" and row["dark"]))


def pick_traits(species=None):
    kinds = SPECIES_KINDS.get(species, DEFAULT_KINDS)
    labels = SPECIES_SIZES.get(species)
    population = _roll(POPULATIONS)
    body_plan = _roll(BODY_PLANS, lambda row: row["kind"] in kinds)
    composition = _composition(
        _roll(COMPOSITIONS,
              lambda row: population["group"] or not row["group_only"],
              lambda row: _composition_weight(row, species, body_plan)),
        body_plan)
    return {
        "danger": _roll(DANGERS)["label"],
        "population": population,
        "body_plan": body_plan,
        "size": _roll(SIZES, lambda row: not labels or row["label"] in labels),
        "palette": _roll(_for_kind(PALETTES, body_plan["kind"]),
                         lambda row: _palette_fits(row, composition)),
        "surface": _roll(_for_kind(SURFACES, body_plan["kind"])),
        "register": _roll(REGISTERS),
        "composition": composition,
    }


_DEFAULT_TRAITS = {
    "composition": lambda: _composition(COMPOSITIONS[0], BODY_PLANS[0]),
    "body_plan": lambda: BODY_PLANS[0],
    "size": lambda: SIZES[2],
    "palette": lambda: PALETTES[0],
    "surface": lambda: SURFACES[0],
    "register": lambda: REGISTERS[0],
    "population": lambda: POPULATIONS[1],
}


def filled(traits):
    traits = dict(traits or {})
    for key, default in _DEFAULT_TRAITS.items():
        if not traits.get(key):
            traits[key] = default()
    return traits


def draws_group(traits):
    traits = filled(traits)
    return traits["population"]["group"] and traits["composition"]["shows_group"]


def draws_foreground_group(traits):
    traits = filled(traits)
    return (draws_group(traits) and not traits["composition"]["group_only"]
            and not traits["body_plan"]["colonial"])


def draws_solo(traits):
    traits = filled(traits)
    return (traits["composition"]["solo"] and not draws_group(traits)
            and not traits["body_plan"]["colonial"])


LIMBLESS_NEGATIVE = "legs, arms, paws, claws, hooves, standing on legs, walking"


def extra_negative(traits):
    traits = filled(traits)
    plan = traits["body_plan"]
    heads = (traits["composition"]["negative"], plan["negative"],
             LIMBLESS_NEGATIVE if plan["limbs"] == "none" else "")
    parts = [part.strip() for head in heads for part in head.split(",") if part.strip()]
    return ", ".join(dict.fromkeys(parts))


def _strip_article(phrase):
    key = " ".join(phrase.split()).lower()
    for article in ("the ", "a ", "an ", "one "):
        if key.startswith(article):
            return key[len(article):]
    return key


def _echoed(head, body):
    haystack = " ".join(body.split()).lower()
    parts = [_strip_article(part) for part in head.split(",")]
    parts = [part for part in parts if len(part) >= 12]
    if not parts:
        return _strip_article(head) in haystack
    hits = sum(1 for part in parts if part in haystack)
    return hits >= min(2, len(parts))


def _with_directives(prompt, composition, foreground_group, body_plan=None):
    body = prompt.strip()
    heads = [composition["directive"]]
    if body_plan is not None and composition["draws_creature"]:
        heads.append(body_plan["en"])
    if foreground_group:
        heads.append(GROUP_DIRECTIVE)
    for head in reversed(heads):
        if not _echoed(head, body):
            body = f"{head}, {body}"
    return body


def generate_profile(target, description, traits):
    traits = filled(traits)
    plan, size = traits["body_plan"], traits["size"]
    palette, surface = traits["palette"], traits["surface"]
    sample_target, sample_body, sample_sheet, _ = _samples(plan)
    if plan["limbs"] == "count":
        sample_traits = ("人間への危険度は「無害。刺激しても壁の隙間へ逃げ込むのみ」、"
                         "個体数は「記録が数例しかない希少種」としてください。"
                         "体長の行は数センチ程度の大きさにし、体色と質感の行は、"
                         "岩や灰に似た灰色を基調とし、薄い粘液に覆われてぬめるものとして書いてください。")
        cap = min(plan["limb_cap"], MAX_LIMBS)
        limb_rule = (
            "付属肢の行には、脚・腕・触手・翼・ひれ・触角の本数を必ず算用数字で書き、片側何本かも添えてください。"
            "「多数」「無数」のような曖昧な書き方はせず、無い付属肢は書かないでください。"
            f"同じ種類の付属肢は多くても{cap}本までとし、図版で数えて確かめられる本数に収めてください。"
        )
    else:
        sample_traits = ("人間への危険度は「飲み込むと体内に寄生する。触れるだけなら害はない」、"
                         "個体数は「生息地では普通に見られる」としてください。"
                         "体長の行は数十センチ程度の大きさにし、体色と質感の行は、"
                         "色素がほとんど無く半透明で、薄い粘液に覆われてぬめるものとして書いてください。")
        limb_rule = (
            "この体のつくりに脚・腕・触手・翼・ひれ・触角はありません。"
            "付属肢の行は「付属肢は無い。」から始め、そのあとに体を支えたり位置を変えたりしている"
            "部分のかたちだけを書いてください。本数は書かないでください。"
        )
    messages = [
        {
            "role": "user",
            "content": (
                f"{sample_target}は以下のような生物です。\n\n{sample_body}\n\n"
                "この生物の、図鑑の解説文には載せない裏設定を作ってください。"
                f"{sample_traits}"
            ),
        },
        {"role": "assistant", "content": sample_sheet},
        {
            "role": "user",
            "content": (
                f"いいですね。次は{target}です。以下のような生物です。\n\n{description}\n\n"
                "この生物の、図鑑の解説文には載せない裏設定を同じ形式で作ってください。"
                "体長 / 体色と質感 / 頭部 / 付属肢 / 特徴的な器官 / 食性 / 人間への危険度 / 個体数 / "
                "行動と姿勢 / 生息環境の細部 の10項目を、この順番で1行ずつ、markdown等は使わずに書いてください。\n\n"
                f"この生物の体のつくりは「{plan['label']}」——{plan['ja']}——です。"
                "頭部・付属肢・体色と質感・行動と姿勢の各行は、この体のつくりから外れないように書いてください。\n\n"
                f"{limb_rule}\n\n"
                f"体長の行は{size['ja']}の大きさにし、体色と質感の行は、"
                f"{palette['ja']}、{surface['ja']}ものとして書いてください。"
                "ただし説明文が大きさや色に触れている場合は、そちらに合わせてください。\n\n"
                f"人間への危険度は「{traits['danger']}」、個体数は「{traits['population']['label']}」として、"
                "それに合う姿・行動にしてください。"
                "上の説明文と矛盾しない範囲で、説明文には書かれていない見た目の細部を補ってください。"
                "項目のみを答え、前置きや解説はしないでください。"
            ),
        },
    ]
    return call_llm(messages)


def generate_scientific_name(target, description):
    messages = [
        {
            "role": "user",
            "content": "古代の湖にて観測される甲殻類「ブリリア」について教えて下さい。markdown等は使用せず文章のみで回答してください"
        },
        {
            "role": "assistant",
            "content": EsukaKnight
        },
        {
            "role": "user",
            "content": "古代の湖にて観測される甲殻類「ブリリア」の学名を考えてください。2単語で。見た瞬間に意味がわかるようなわかりやすいものは避けてください。学名のみを答えてください"
        },
        {
            "role": "assistant",
            "content": "Testaceobrachia propulsus"
        },
        {
            "role": "user",
            "content": "いいですね。次はアンカラ洞窟にて観測される架空の生物「ザトン」について教えて下さい。markdown等は使用せず文章のみで回答してください"
        },
        {
            "role": "assistant",
            "content": Kyomuton
        },
        {
            "role": "user",
            "content": "アンカラ洞窟にて観測される架空の生物「ザトン」の学名を考えてください。2単語で。見た瞬間に意味がわかるようなわかりやすいものは避けてください。学名のみを答えてください"
        },
        {
            "role": "assistant",
            "content": "Spelaeoneura parietalis"
        },
        {
            "role": "user",
            "content": f"素晴らしいですね。次は{target}について教えて下さい。markdown等は使用せず文章のみで回答してください"
        },
        {
            "role": "assistant",
            "content": description
        },
        {
            "role": "user",
            "content": f"{target}の学名を考えてください。2単語で。見た瞬間に意味がわかるようなわかりやすいものは避けてください。学名のみを答えてください"
        },
    ]
    return first_line(call_llm(messages))


def _samples(plan):
    if plan["limbs"] == "count":
        return _ZATON, Kyomuton, KyomutonProfile, KyomutonPrompt
    return _MIZU, Mizumodoki, MizumodokiProfile, MizumodokiPrompt


def _subject_note(composition, foreground_group, plan):
    if foreground_group:
        return "画面には同じ種の個体が多数写りますが、本数を数えられるのは手前の一体だけで構いません。"
    if not composition["solo"] and not composition["group_only"]:
        return "描くのは幼体と成体の二体だけで、どちらも同じ本数にしてください。"
    if plan["colonial"]:
        return ("描くのは一つの群体だけですが、それが同じ小さな単位の集まりでできていること、"
                "一匹の大きな生きもののかたちにはならないことがわかるように書いてください。")
    return "描くのは一体だけにしてください。"


def generate_prompt(target, description, profile="", traits=None):
    traits = filled(traits)
    composition, plan, size = traits["composition"], traits["body_plan"], traits["size"]
    palette, surface = traits["palette"], traits["surface"]
    foreground_group = draws_foreground_group(traits)
    subject_note = _subject_note(composition, foreground_group, plan)
    sample_target, sample_body, sample_sheet, sample_prompt = _samples(plan)
    if composition["limb_mode"] == "count" and plan["limbs"] == "none":
        limb_note = (
            "この生物に脚・腕・触手・翼・ひれ・触角はありません。本数は書かず、"
            "no legs, no arms, no tentacles, no other limbs と添えてください。"
            "体を支えている部分のかたちだけを書いてください。" + subject_note
        )
    elif composition["limb_mode"] == "count":
        limb_note = (
            "脚・触手・腕・翼・ひれ・触角といった付属肢は、裏設定に書かれた本数どおりに、"
            "必ず英語の数詞で書いてください（例: six thin twitching legs, three legs on each side）。"
            "many legs, numerous limbs, multiple tentacles のような数の曖昧な表現は使わず、"
            "同じ部位の本数を別の箇所で違う数で書かないでください。"
            "裏設定に無い種類の付属肢は生やさないよう no other limbs と添えてください。"
            + subject_note
        )
    elif composition["limb_mode"] == "distant":
        limb_note = (
            "この構図では個体は遠くに小さくしか写りません。付属肢の本数や体の細部は書かず、"
            "遠くから見た体のかたちと色、群れ全体の広がり・密度・分布のかたち、"
            "そして周囲の生息環境を描写してください。"
        )
    else:
        limb_note = (
            "この構図では生物の本体は画面に登場しません。体の描写は書かず、"
            "その生物が残した痕跡と、それが残っている生息環境だけを描写してください。"
            "no creature visible と添えてください。"
        )
    profile_block = (
        f"さらに、図鑑には載せていない裏設定が以下のとおりです。プロンプトの細部はここから取ってください。"
        f"\n\n{profile}\n\n" if profile.strip() else ""
    )
    notes = []
    if composition["draws_creature"]:
        notes.append(
            f"この生物の体のつくりは「{plan['label']}」——{plan['ja']}——で、"
            f"英語では次のように書きます: {plan['en']}。この形から外れる描写はしないでください。")
        notes.append(
            f"体色と体表は英語では次のように書きます: {palette['en']}, {surface['en']}。"
            f"この色と質感は必ずプロンプトに入れ、ほかの色で塗り替えないでください。")
    if not composition["magnifies_body"]:
        scale_tail = ("周囲のものとの対比でその大きさが伝わるように書いてください。"
                      if composition["draws_creature"]
                      else "残された痕跡の大きさがこれに見合うように書いてください。")
        notes.append(
            f"この生物の大きさは{size['ja']}で、英語では {size['en']} と書きます。{scale_tail}"
            "説明文や裏設定がこれと違う大きさを書いている場合は、そちらに合わせてください。")
    if foreground_group:
        notes.append(
            f"この生物は群れをつくり、画面には同じ姿の個体が多数写ります。英語では次のように指定されています: "
            f"{GROUP_DIRECTIVE}。群れの密度や、集まっているときの行動が伝わる描写を入れてください。"
            f"ただし手前の一体は全身がはっきり見えるように書いてください。")
    notes.append(limb_note)
    notes_block = "".join(f"{note}\n\n" for note in notes)
    messages = [
        {
            "role": "user",
            "content": (
                f"{sample_target}は以下のような生物です。\n\n{sample_body}\n\n"
                f"さらに、図鑑には載せていない裏設定が以下のとおりです。\n\n{sample_sheet}\n\n"
                f"そしてこの生物のイメージを描くためのプロンプトが以下のとおりです。\n\n{sample_prompt}\n\n"
                f"これにならって、以下のような{target}のイメージを描くためのプロンプトを英語で作成してください。"
                f"\n\n{description}\n\n{profile_block}"
                f"この絵の構図は「{composition['label']}」で、英語では次のように指定されています: "
                f"{composition['directive']}。この構図に合う内容だけを書いてください。\n\n"
                f"{notes_block}"
                "プロンプトのみを答え、解説等はしないでください。あなたの出力はそのままStable Diffusionに渡されます。\n\n"
                "画風・画質・照明の指定は別途こちらで付与するので、あなたは生物の形態と生息環境の描写だけを書いてください。"
                "masterpiece, best quality, absurdres, 8k, ultra-detailed のような品質タグや、"
                "cinematic, dramatic lighting, volumetric lighting, glowing のような演出タグは一切使わないでください"
            ),
        }
    ]
    return _with_directives(call_llm(messages), composition, foreground_group, plan)


REVIEW_MAX_TOKENS = 512
_REVIEW_LINE = re.compile(r"^\s*(\d+)\s*[.:：]?\s*(OK|NG)\b[\s:：ー-]*(.*)$", re.IGNORECASE)
_REVIEW_WORDS = re.compile(r"[^A-Za-z0-9 ,'-]+")
_REVIEW_NEGATIONS = ("no", "not", "without", "missing", "lack", "lacking",
                     "absent", "zero", "none", "fewer", "less")


def _review_items(prompt, traits):
    composition, plan = traits["composition"], traits["body_plan"]
    items = []
    if composition["draws_creature"]:
        items.append({"key": "body_plan",
                      "label": f"体のつくりが「{plan['ja']}」であること",
                      "fix": plan["en"]})
        if composition["limb_mode"] == "count":
            if plan["limbs"] == "none":
                items.append({"key": "limbs",
                              "label": "脚・腕・触手・翼・ひれ・触角が生えていないこと",
                              "fix": ""})
            else:
                counts = limb_count_phrases(prompt)
                if counts:
                    items.append({"key": "limbs",
                                  "label": "付属肢の本数と左右の分かれ方が「"
                                           + "」「".join(counts) + "」であること",
                                  "fix": ", ".join(counts)})
    else:
        items.append({"key": "absent",
                      "label": "生物の本体が画面に写っておらず、残された痕跡と生息環境だけが写っていること",
                      "fix": "no creature visible"})
    items.append({"key": "composition",
                  "label": f"構図が「{composition['label']}」——{composition['directive']}——であること",
                  "fix": composition["directive"]})
    if composition["draws_creature"]:
        if draws_foreground_group(traits):
            items.append({"key": "group",
                          "label": "同じ姿の個体が多数写っていること",
                          "fix": GROUP_DIRECTIVE})
        elif draws_solo(traits):
            items.append({"key": "solo",
                          "label": "写っている個体が一体だけであること",
                          "fix": ""})
        items.append({"key": "body",
                      "label": "体の色・かたち・部位・大きさが説明文や裏設定と矛盾していないこと",
                      "fix": ""})
    items.append({"key": "habitat",
                  "label": "写っている場所が説明文の生息環境と矛盾していないこと",
                  "fix": ""})
    return items


def _review_note(text):
    notes = []
    for part in _REVIEW_WORDS.sub(" ", text).split(","):
        part = " ".join(part.split()).strip(" -'")
        words = part.lower().split()
        if not words or len(words) > 6 or words[0] in _REVIEW_NEGATIONS:
            continue
        if part.lower() not in [note.lower() for note in notes]:
            notes.append(part)
    return ", ".join(notes[:2])


def _parse_review(text, items):
    verdicts = {}
    for line in text.splitlines():
        match = _REVIEW_LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < len(items):
            verdicts[index] = (match.group(2).upper() == "OK", _review_note(match.group(3)))
    review = []
    for index, item in enumerate(items):
        ok, note = verdicts.get(index, (True, ""))
        review.append({"key": item["key"], "label": item["label"], "fix": item["fix"],
                       "ok": ok, "note": note})
    return review


def review_image(image, prompt, target, description, profile="", traits=None):
    traits = filled(traits)
    items = _review_items(prompt, traits)
    checklist = "\n".join(f"{index + 1}. {item['label']}" for index, item in enumerate(items))
    profile_block = f"図鑑には載せていない裏設定: {profile}\n\n" if profile.strip() else ""
    buf = BytesIO()
    image.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": (
                        f"これは「{target}」の図版として生成した画像です。\n\n"
                        f"生物の説明: {description}\n\n"
                        f"{profile_block}"
                        f"作画に使った指示: {prompt}\n\n"
                        "この画像が以下の各項目と食い違っていないかを判定してください。\n\n"
                        f"{checklist}\n\n"
                        "判定の規則:\n"
                        "・画像から判別できない項目、小さすぎて読み取れない項目は OK としてください。"
                        "描かれていないことは食い違いではありません。はっきり矛盾しているときだけ NG です。\n"
                        f"・1行に1項目、1から{len(items)}まで順に、"
                        "「番号: OK」または「番号: NG 語句」の形式で書いてください。\n"
                        "・NG の語句には、矛盾の原因として画像に実際に写っているものを英語で書いてください"
                        "（例:「2: NG segmented exoskeleton, jointed legs」）。"
                        "3語以内の語句をカンマ区切りで多くても2つ、そのまま negative prompt に渡します。\n"
                        "・判定の行以外は何も書かないでください。"
                    ),
                },
            ],
        }
    ]
    return _parse_review(call_llm(messages, max_tokens=REVIEW_MAX_TOKENS), items)


def _asks_for(note, wanted):
    words = [word for word in re.findall(r"[a-z0-9]+", note.lower()) if len(word) > 2]
    return bool(words) and all(word in wanted for word in words)


def review_payload(review):
    return [{"key": entry["key"], "ok": entry["ok"], "note": entry["note"]} for entry in review]


def _promote(prompt, phrase):
    at = prompt.lower().find(phrase.lower())
    if at >= 0:
        prompt = prompt[:at] + prompt[at + len(phrase):]
    prompt = re.sub(r"\s*,(\s*,)+", ",", prompt).strip().strip(",").strip()
    return f"{phrase}, {prompt}"


def apply_review(prompt, review):
    for entry in reversed(review):
        if not entry["ok"] and entry["fix"]:
            prompt = _promote(prompt, entry["fix"])
    wanted = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    notes = []
    for entry in review:
        if entry["ok"]:
            continue
        for note in entry["note"].split(","):
            note = note.strip()
            if not note or _asks_for(note, wanted):
                continue
            if note.lower() not in [kept.lower() for kept in notes]:
                notes.append(note)
    return prompt, ", ".join(notes)


PROOF_MAX_TOKENS = 1024
_SENTENCE = re.compile(r"[^。\n]+。?")


def _sentences(description):
    return [match for match in _SENTENCE.finditer(description) if match.group().strip()]


def _as_sentence(text):
    text = " ".join(text.split()).strip().strip("「」『』\"'")
    if text and not text.endswith("。"):
        text += "。"
    return text


def _keeps_words(text, original, forbidden):
    return (all(not word or word not in original or word in text for word in forbidden["keep"])
            and all(not word or word not in text for word in forbidden["hidden"]))


def _revised_sentence(text, original, forbidden):
    text = _as_sentence(text)
    if not text or len(text) > len(original) * 1.5 + 10:
        return ""
    if not _keeps_words(text, original, forbidden):
        return ""
    return "" if text == original else text


def _numbered(sentences):
    return "\n".join(f"{index + 1}. {match.group().strip()}"
                     for index, match in enumerate(sentences))


def _sentence_pass(text, sentences, revise, forbidden, max_edits):
    revised, shifted = {}, False
    for line in text.splitlines():
        match = _REVIEW_LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if 0 < index < len(sentences) and match.group(2).upper() == "NG":
            fixed = revise(match.group(3), sentences[index].group().strip(), forbidden)
            shifted = shifted or any(fixed == other.group().strip() for other in sentences)
            if fixed:
                revised[index] = fixed
    if shifted or len(revised) > max_edits:
        revised = {}
    return [{"span": match.span(), "sentence": match.group().strip(),
             "ok": index not in revised, "text": revised.get(index, "")}
            for index, match in enumerate(sentences)]


def _proof_forbidden(species, field, traits):
    plan = traits["body_plan"]
    return {"keep": (species, field),
            "hidden": (traits["population"]["label"], traits.get("danger", ""),
                       plan["label"], plan["ja"])}


def review_description(image, description, target, species="", field="", traits=None):
    traits = filled(traits)
    sentences = _sentences(description)
    forbidden = _proof_forbidden(species, field, traits)
    composition = traits["composition"]
    if not composition["draws_creature"]:
        return _sentence_pass("", sentences, _revised_sentence, forbidden, 1)
    plan = traits["body_plan"]
    distant_rule = ("・この図版では個体は遠くに小さくしか写りません。"
                    "本数・器官・質感といった体の細部は判断できないものとして扱ってください。\n"
                    if composition["limb_mode"] == "distant" else "")
    buf = BytesIO()
    image.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": (
                        f"これは「{target}」の図鑑に載せる図版です。\n\n"
                        "同じ図鑑に載せる解説文を1文ずつ並べます。"
                        "図版と読み比べて、図版にはっきり写っていることと食い違う文だけを書き直してください。\n\n"
                        f"{_numbered(sentences)}\n\n"
                        "判定の規則:\n"
                        f"・この生物が{field}に生息する{species}であること、"
                        f"体のつくりが「{plan['ja']}」であること、名前と学名は、"
                        "どれも正しいものとして扱ってください。"
                        "図版がこれらと食い違って見えても、直すのは図版であって解説文ではありません。\n"
                        "・図版から判断できない事柄——生態、行動、季節、繁殖、伝承、匂い、内部の構造、"
                        "小さすぎる部分や画面の外にある部分——は、そのままで正しいものとします。\n"
                        f"{distant_rule}"
                        "・ほとんどの文はそのままのはずです。書き直すのは、"
                        "この図版を見た人が明らかにおかしいと気づく記述だけにしてください。\n"
                        f"・1行に1文、1から{len(sentences)}まで順に、"
                        "「番号: OK」または「番号: NG 書き直した文」の形式で書いてください。\n"
                        "・書き直す文は、その番号の文を図版に合うように直した1文だけを書いてください。"
                        "他の文の内容を持ち込まず、元の文と同じ話題・同じ文体・同程度の長さを保ってください。\n"
                        "・一文目はその生物が何であるかを言い切る文なので、常に OK としてください。\n"
                        f"・{traits['register']['instruction']}\n"
                        "・判定の行以外は何も書かないでください。"
                    ),
                },
            ],
        }
    ]
    return _sentence_pass(call_llm(messages, max_tokens=PROOF_MAX_TOKENS), sentences,
                          _revised_sentence, forbidden, 1)


def apply_proof(description, review):
    for entry in reversed(review):
        if entry["ok"]:
            continue
        start, end = entry["span"]
        description = description[:start] + entry["text"] + description[end:]
    return description


POLISH_MAX_TOKENS = 1024
POLISH_MAX_EDITS = 2
_NUMERALS = re.compile(r"[0-9０-９一二三四五六七八九十百千万]+")
_KANJI = re.compile(r"[\u4e00-\u9fff]")


def _polish_forbidden(description, name, species, field, traits):
    forbidden = _proof_forbidden(species, field, traits)
    return {"keep": forbidden["keep"] + (name,), "hidden": forbidden["hidden"],
            "source": description}


def _natural_sentence(text, original, forbidden):
    text = _as_sentence(text)
    if not text or Levenshtein.distance(text, original) > max(6, len(original) // 3):
        return ""
    if len(original) - len(text) > 6:
        return ""
    if len(original) - len(os.path.commonprefix([original, text])) <= 6:
        return ""
    if set(_KANJI.findall(text)) - set(_KANJI.findall(forbidden["source"])):
        return ""
    if _NUMERALS.findall(text) != _NUMERALS.findall(original):
        return ""
    if not _keeps_words(text, original, forbidden):
        return ""
    return "" if text == original else text


def polish_description(description, name="", species="", field="", traits=None):
    traits = filled(traits)
    sentences = _sentences(description)
    forbidden = _polish_forbidden(description, name, species, field, traits)
    messages = [
        {
            "role": "user",
            "content": (
                "図鑑に載せる解説文を1文ずつ並べます。"
                "日本語として不自然なところだけを直してください。\n\n"
                f"{_numbered(sentences)}\n\n"
                "直すのは次のような場合だけです:\n"
                "・助詞が誤っている、主語と述語がねじれている、係り受けが通っていない\n"
                "・動詞の活用や自動詞・他動詞の使い方が誤っている\n"
                "・語の組み合わせが日本語として意味を成していない\n\n"
                "次は誤りではありません。OK としてください:\n"
                "・文末が「する」か「している」か、助詞が「に」か「へ」かといった書き分け\n"
                "・一文目の「——する生物。」のような体言止め\n"
                "・硬い言い回し、まわりくどい言い回し、説明の順序\n\n"
                "次のものは直さないでください:\n"
                "・書かれている内容。事実・数値・生物の名前・生息地や分類の呼び名は変えないでください。\n"
                f"・文体と語調。{traits['register']['instruction']}\n"
                "・文の数。1文を2文に分けたり、2文をまとめたりしないでください。\n"
                "・言い回しの好み。読んで意味が通る文は、硬くても回りくどくても OK としてください。\n\n"
                f"・1行に1文、1から{len(sentences)}まで順に、"
                "「番号: OK」または「番号: NG 直した文」の形式で書いてください。\n"
                "・直す文は、元の文の語をできるだけ残したまま、"
                "おかしい箇所だけを最小限に書き換えた1文を書いてください。\n"
                "・ほとんどの文は OK のはずです。\n"
                "・判定の行以外は何も書かないでください。"
            ),
        }
    ]
    return _sentence_pass(call_llm(messages, max_tokens=POLISH_MAX_TOKENS), sentences,
                          _natural_sentence, forbidden, POLISH_MAX_EDITS)


def _description_request(target, ja, register, opener=""):
    return (f"{opener}{target}について3から5文程度で教えて下さい。{_OPENING}"
            f"この生物の姿は{ja}です。"
            f"その体で何をして暮らしているかが伝わるように書いてください。{register['instruction']}"
            "markdown等は使用せず文章のみで回答してください")


def generate_description(target, traits=None):
    traits = filled(traits)
    plan, register = traits["body_plan"], traits["register"]
    messages = []
    for phrase, text in register["samples"]:
        sample_plan = _PLAN_LABELS[_SAMPLE_PLANS[phrase]]
        messages.append({"role": "user",
                         "content": _description_request(phrase, sample_plan["ja"], register)})
        messages.append({"role": "assistant", "content": text})
    messages.append({"role": "user",
                     "content": _description_request(target, plan["ja"], register, "いいですね。次は")})
    return call_llm(messages)
