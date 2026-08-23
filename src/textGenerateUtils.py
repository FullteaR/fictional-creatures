import os
import base64
import random
import re
from io import BytesIO
from openai import OpenAI
from sampleMonsters import *

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080/v1")
_client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="dummy")
_MODEL = "local-model"



def call_llm(messages, max_tokens=2048):
    # max_tokens は必須。無いと Q2_K_XL が繰り返しループに落ちたときに 32k の
    # コンテキストを使い切るまで走り続け、llama-server が OOM で kill される。
    # temperature / top_p / top_k / min_p は llama-server 側 (Dockerfile) で指定。
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
            print(delta, end="", flush=True)
            chunks.append(delta)
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    print()
    if finish_reason == "length":
        print(f"[warn] max_tokens={max_tokens} に到達して打ち切られました（繰り返しループの可能性）")
    return "".join(chunks)



def first_line(text):
    """複数行で返ってきた場合に先頭の非空行だけを採用する"""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


# 拡散モデルは「脚の数」を勝手に決めるので、プロンプト側で数詞を書かせて固定する。
# 下書き → 画像リファインの2パスで本数がずれないよう、下書きの数詞を抽出して次のパスに渡す。
_NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
# forelimbs / hindlegs / walking legs / chelipeds まで拾えないと、
# 「four forelimbs」を取りこぼして本数の引き継ぎが効かない
_LIMB_PART = (r"(?:(?:fore|hind|mid|middle|walking|swimming)[- ]?)?"
              r"(?:legs?|limbs?|arms?|tentacles?|wings?|fins?|claws?|pincers?|antennae|chelipeds?|appendages?)")
_LIMB_COUNT_RE = re.compile(
    rf"\b{_NUMBER}(?:\s+pairs?\s+of)?(?:\s+[a-z-]+){{0,3}}\s+{_LIMB_PART}(?:\s+on each side)?\b",
    re.IGNORECASE,
)


def limb_count_phrases(prompt):
    """プロンプト中の「数詞 + 付属肢」表現を拾う (例: "six slender legs")"""
    seen = []
    for match in _LIMB_COUNT_RE.findall(prompt):
        phrase = " ".join(match.split())
        if phrase.lower() not in [s.lower() for s in seen]:
            seen.append(phrase)
    return seen


# ---- 裏設定 ---------------------------------------------------------------
# 図鑑の解説文（カードに載る文章）には出さないが、絵作りの材料にする設定。
# 危険度・個体数・構図はコード側でサイコロを振る。LLM に選ばせると
# どれも「無害・希少・生物の全体図」に寄って 25 枚が同じ絵面になるため。

# 構図は STYLE_PREFIX と同じ扱いで、LLM の出力にコード側から前置きする。
# draws_creature=False の構図では本体を描かないので、付属肢の本数指定も外す。
# 最後の要素は構図ごとの追加ネガティブ。特に「痕跡」は、プロンプトで no creature visible と
# 言うだけでは本体が描かれてしまうので、ネガティブ側から生きた個体を潰す必要がある。
COMPOSITIONS = [
    (50, "生物の全体図", "the whole creature centred in frame, entire body visible from head to tail, "
                        "side-on specimen view, habitat kept plain and secondary", True,
     "cropped, out of frame, extreme close-up"),
    (20, "生息地の風景", "wide view of the habitat filling the frame, the creature small and partly "
                        "concealed within the scene, environment shown in full", True,
     "extreme close-up, empty scenery"),
    (15, "生態の痕跡", "the creature itself absent from frame, no animal visible, only {trace} "
                      "left behind, shown in situ in the empty habitat", False,
     "live animal, living creature, animal, eyes, face, moving limbs"),
    (15, "体の一部の拡大図", "close-up study of {part} filling the frame, the rest of the body out of "
                          "frame, habitat plain and out of focus behind", True,
     "full body, whole creature, wide shot, distant view"),
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
    "the head", "one limb", "the mouthparts",
    "the patterned surface of the body", "the sensory organs",
]
DANGERS = [
    (25, "人間には全く無害"),
    (35, "刺激すると刺す、あるいは咬む程度"),
    (25, "毒を持ち、接触すると危険"),
    (15, "致死的で、接近そのものが極めて危険"),
]
# 個体数は絵に直接出しにくいが、行動や生息環境の描写を通して間接的に効く。
# 「大量発生」でも図版に描くのは一体だけ。複数個体は NEGATIVE_PROMPT 側で落としており、
# 画面内に何匹も入ると脚の本数が読めなくなる。
POPULATIONS = [
    (20, "大量発生しており、生息地では群れに出くわす"),
    (30, "生息地では普通に見られる"),
    (30, "限られた場所にのみ局所的に生息する"),
    (20, "記録が数例しかない希少種"),
]


def _composition(row):
    _, label, directive, draws_creature, negative = row
    return {
        "label": label,
        "directive": directive.format(trace=random.choice(TRACES), part=random.choice(PARTS)),
        "draws_creature": draws_creature,
        "negative": negative,
    }


def pick_traits():
    """コード側でサイコロを振る裏設定。図鑑の説明文には出さない"""
    return {
        "danger": random.choices([d[1] for d in DANGERS], weights=[d[0] for d in DANGERS])[0],
        "population": random.choices([p[1] for p in POPULATIONS], weights=[p[0] for p in POPULATIONS])[0],
        "composition": _composition(random.choices(
            COMPOSITIONS, weights=[row[0] for row in COMPOSITIONS])[0]),
    }


def _composition_of(traits):
    if traits and traits.get("composition"):
        return traits["composition"]
    return _composition(COMPOSITIONS[0])


def _with_composition(prompt, composition):
    """構図指定はコード側で前置きする（LLM に任せると毎回「全体図」に戻る）"""
    body = prompt.strip()
    head = composition["directive"]
    if head.split(",")[0].strip().lower() in body.lower():
        return body
    return f"{head}, {body}"


def generate_profile(target, description, traits):
    """図鑑の解説文には載せない裏設定シートを作る。プロンプト生成の材料にする"""
    messages = [
        {
            "role": "user",
            "content": (
                f"アンカラ洞窟にて観測される架空の生物「ザトン」は以下のような生物です。\n\n{Kyomuton}\n\n"
                "この生物の、図鑑の解説文には載せない裏設定を作ってください。"
                "人間への危険度は「無害。刺激しても壁の隙間へ逃げ込むのみ」、"
                "個体数は「記録が数例しかない希少種」としてください。"
            ),
        },
        {"role": "assistant", "content": KyomutonProfile},
        {
            "role": "user",
            "content": (
                f"いいですね。次は{target}です。以下のような生物です。\n\n{description}\n\n"
                "この生物の、図鑑の解説文には載せない裏設定を同じ形式で作ってください。"
                "体長 / 体色と質感 / 頭部 / 付属肢 / 特徴的な器官 / 食性 / 人間への危険度 / 個体数 / "
                "行動と姿勢 / 生息環境の細部 の10項目を、この順番で1行ずつ、markdown等は使わずに書いてください。\n\n"
                "付属肢の行には、脚・腕・触手・翼・ひれ・触角の本数を必ず算用数字で書き、片側何本かも添えてください。"
                "「多数」「無数」のような曖昧な書き方はせず、無い付属肢は書かないでください。\n\n"
                f"人間への危険度は「{traits['danger']}」、個体数は「{traits['population']}」として、"
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
    # 学名は2単語固定。万一複数行で返ってきても先頭行だけ採る
    # (長文が混ざると add_caption の max_line_width を支配してレイアウトが壊れる)
    return first_line(call_llm(messages))


def generate_prompt(target, description, profile="", traits=None):
    composition = _composition_of(traits)
    sample_prompt = """
    intricate cave system, vast underground cavern, damp misty air, rugged cave walls, uneven rocky surfaces, small flying insects, pale fungi, scattered moss patches, hanging roots, water droplets on stalactites, cold humid air, ancient untouched cave, a tiny creature clinging to the wall, large round eyes, six thin twitching legs, three legs on each side, two short antennae, no other limbs, one specimen shown clearly, whole body visible, poised to strike at a passing insect, delicate organism against rough stone, plain background, specimen study
    """
    if composition["draws_creature"]:
        limb_note = (
            "脚・触手・腕・翼・ひれ・触角といった付属肢は、裏設定に書かれた本数どおりに、"
            "必ず英語の数詞で書いてください（例: six thin twitching legs, three legs on each side）。"
            "many legs, numerous limbs, multiple tentacles のような数の曖昧な表現は使わず、"
            "同じ部位の本数を別の箇所で違う数で書かないでください。"
            "裏設定に無い種類の付属肢は生やさないよう no other limbs と添え、描くのは一体だけにしてください。"
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
    messages = [
        {
            "role": "user",
            "content": (
                f"アンカラ洞窟にて観測される架空の生物「ザトン」は以下のような生物です。\n\n{Kyomuton}\n\n"
                f"さらに、図鑑には載せていない裏設定が以下のとおりです。\n\n{KyomutonProfile}\n\n"
                f"そしてこの生物のイメージを描くためのプロンプトが以下のとおりです。\n\n{sample_prompt}\n\n"
                f"これにならって、以下のような{target}のイメージを描くためのプロンプトを英語で作成してください。"
                f"\n\n{description}\n\n{profile_block}"
                f"この絵の構図は「{composition['label']}」で、英語では次のように指定されています: "
                f"{composition['directive']}。この構図に合う内容だけを書いてください。\n\n"
                f"{limb_note}\n\n"
                "プロンプトのみを答え、解説等はしないでください。あなたの出力はそのままStable Diffusionに渡されます。\n\n"
                "画風・画質・照明の指定は別途こちらで付与するので、あなたは生物の形態と生息環境の描写だけを書いてください。"
                "masterpiece, best quality, absurdres, 8k, ultra-detailed のような品質タグや、"
                "cinematic, dramatic lighting, volumetric lighting, glowing のような演出タグは一切使わないでください"
            ),
        }
    ]
    return _with_composition(call_llm(messages), composition)


def refine_prompt_with_image(prompt, image, target, description, profile="", traits=None):
    """生成画像を踏まえてプロンプトを改善する。"""
    composition = _composition_of(traits)
    # 下書きの本数をそのまま次のパスに持ち越す。画像側は脚の数を間違えているのが
    # 常態なので、「画像に合わせる」のではなく下書きの数詞を正としてもう一度書かせる。
    counts = limb_count_phrases(prompt) if composition["draws_creature"] else []
    if counts:
        limb_note = (
            "元のプロンプトでは付属肢の本数を「" + "」「".join(counts) + "」と指定しています。"
            "画像に写っている本数がこれと違っていても、正しいのは元のプロンプトの方です。"
            "書き直したプロンプトにも同じ数詞をそのまま書いてください。"
            "many legs, numerous limbs のような数の曖昧な表現は使わず、"
            "裏設定に無い種類の付属肢が生えないよう no other limbs と添え、描くのは一体だけにしてください。"
        )
    elif composition["draws_creature"]:
        limb_note = (
            "書き直したプロンプトでは、脚・触手・腕・翼・ひれ・触角の本数を裏設定どおりに英語の数詞で書いてください"
            "（例: six thin legs, three legs on each side）。描くのは一体だけにしてください。"
        )
    else:
        limb_note = (
            "この構図では生物の本体は画面に登場しません。体の描写は書かず、痕跡と生息環境だけを描写し、"
            "no creature visible と添えてください。"
        )
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
                        f"これは「{target}」を描くために以下のプロンプトで生成した画像です。\n\n"
                        f"プロンプト: {prompt}\n\n"
                        f"生物の説明: {description}\n\n"
                        f"{profile_block}"
                        f"この絵の構図は「{composition['label']}」で、英語では次のように指定されています: "
                        f"{composition['directive']}\n\n"
                        "この画像と生物の説明・裏設定を見比べ、食い違っている点を洗い出したうえで、"
                        "説明と裏設定をより正確に反映するようプロンプトを書き直してください。"
                        "書き直したプロンプトのみを英語で答えてください。解説は不要です。あなたの出力はそのままStable Diffusionに渡されます。\n\n"
                        f"{limb_note}\n\n"
                        "画風・画質・照明の指定はこちら側で別途付与するので、あなたは生物の形態と生息環境の描写だけを書いてください。"
                        "masterpiece, best quality, absurdres, 8k, ultra-detailed のような品質タグや、"
                        "cinematic, dramatic lighting, volumetric lighting, glowing のような演出タグは一切使わないでください。"
                    ),
                },
            ],
        }
    ]
    return _with_composition(call_llm(messages), composition)


def generate_description(target):
    messages = [
        {
            "role": "user",
            "content": "古代の湖にて観測される甲殻類「ブリリア」について3から5文程度で教えて下さい。markdown等は使用せず文章のみで回答してください"
        },
        {
            "role": "assistant",
            "content": EsukaKnight
        },
        {
            "role": "user",
            "content": "いいですね。次は深海にて観測される架空の生物「ミズモドキ」について3から5文程度で教えて下さい。markdown等は使用せず文章のみで回答してください"
        },
        {
            "role": "assistant",
            "content": Mizumodoki
        },
        {
            "role": "user",
            "content": "いいですね。次はアンカラ洞窟にて観測される架空の生物「ザトン」について3から5文程度で教えて下さい。markdown等は使用せず文章のみで回答してください"
        },
        {
            "role": "assistant",
            "content": Kyomuton
        },
        {
            "role": "user",
            "content": f"いいですね。次は{target}について3から5行程度で教えて下さい。markdown等は使用せず文章のみで回答してください"
        }
    ]
    return call_llm(messages)
