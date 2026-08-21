import os
import base64
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


def generate_prompt(target, description):
    sample_prompt = """
    intricate cave system, vast underground cavern, damp misty air, rugged cave walls, uneven rocky surfaces, small flying insects, pale fungi, scattered moss patches, hanging roots, water droplets on stalactites, cold humid air, ancient untouched cave, a tiny creature clinging to the wall, large round eyes, thin twitching limbs, one specimen shown clearly, poised to strike at a passing insect, delicate organism against rough stone, plain background, specimen study
    """
    messages = [
        {
            "role": "user",
            "content": f"アンカラ洞窟にて観測される架空の生物「ザトン」は以下のような生物です。\n\n{Kyomuton}\n\nそしてこの生物のイメージを描くためのプロンプトが以下のとおりです。\n\n{sample_prompt}\n\n。これにならって、以下のような{target}のイメージを描くためのプロンプトを英語で作成してください。\n\n{description}\n\n生物が大型の場合はその動物を中心に、小型の場合は生息地を中心とした絵を描くようにしてください。プロンプトのみを答え、解説等はしないでください。あなたの出力はそのままStable Diffusionに渡されます。\n\n画風・画質・照明の指定は別途こちらで付与するので、あなたは生物の形態と生息環境の描写だけを書いてください。masterpiece, best quality, absurdres, 8k, ultra-detailed のような品質タグや、cinematic, dramatic lighting, volumetric lighting, glowing のような演出タグは一切使わないでください"
        }
    ]
    return call_llm(messages)


def refine_prompt_with_image(prompt, image, target, description):
    """生成画像を踏まえてプロンプトを改善する。"""
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
                        "この画像と生物の説明を見比べ、食い違っている点を洗い出したうえで、"
                        "説明をより正確に反映するようプロンプトを書き直してください。"
                        "書き直したプロンプトのみを英語で答えてください。解説は不要です。あなたの出力はそのままStable Diffusionに渡されます。\n\n"
                        "画風・画質・照明の指定はこちら側で別途付与するので、あなたは生物の形態と生息環境の描写だけを書いてください。"
                        "masterpiece, best quality, absurdres, 8k, ultra-detailed のような品質タグや、"
                        "cinematic, dramatic lighting, volumetric lighting, glowing のような演出タグは一切使わないでください。"
                    ),
                },
            ],
        }
    ]
    return call_llm(messages)


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
