import re
import os
import base64
from io import BytesIO
from openai import OpenAI
from sampleMonsters import *

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080/v1")
_client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="dummy")
_MODEL = "local-model"



def call_llm(messages):
    stream = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        stream=True
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)



def generate_text(messages):
    out1 = call_llm(messages)
    messages.append({"role": "assistant", "content": out1})
    messages.append({
        "role": "user",
        "content": "今の回答を60点として、userの指示に沿った100点の回答をしてください。改善点等の解説はせず、回答のみを答えてください。長さを水増しするのではなく中身の質を上げてください。"
    })
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
    return generate_text(messages)


def generate_prompt(target, description):
    sample_prompt = """
    masterpiece, best quality, absurdres, intricate cave system, vast underground cavern, damp and misty atmosphere, rugged cave walls, uneven rocky surfaces, small flying insects swarming, faint bioluminescent glow, scattered moss patches, luminescent fungi, hanging roots, water droplets falling from stalactites, quiet and eerie ambiance, hidden crevices, ancient untouched cave, faint echoes, cold and humid air, mysterious and unexplored, tiny fast-moving creatures clinging to walls, large round reflective eyes, twitching limbs, blurry movement, sharp focus on one specimen, watchful gaze, poised to strike at passing insect, stark contrast of delicate organism and rough stone, discovery scene, flashlight beam illuminating creature, sense of rare encounter, scientific expedition vibe
    """
    messages = [
        {
            "role": "user",
            "content": f"アンカラ洞窟にて観測される架空の生物「ザトン」は以下のような生物です。\n\n{Kyomuton}\n\nそしてこの生物のイメージを描くためのプロンプトが以下のとおりです。\n\n{sample_prompt}\n\n。これにならって、以下のような{target}のイメージを描くためのプロンプトを英語で作成してください。\n\n{description}\n\n生物が大型の場合はその動物を中心に、小型の場合は生息地を中心とした絵を描くようにしてください。プロンプトのみを答え、解説等はしないでください。あなたの出力はそのままStable Diffusionに渡されます"
        }
    ]
    return generate_text(messages)


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
                        "この画像を見て、生物の説明をより正確に反映するようプロンプトを改善してください。"
                        "改善したプロンプトのみを英語で答えてください。解説は不要です。あなたの出力はそのままStable Diffusionに渡されます。"
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
    return generate_text(messages)
