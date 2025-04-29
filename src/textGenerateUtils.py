from transformers import TextStreamer
from sampleMonsters import *
import gc
import torch

def generate_scientific_name(target, description, model, tokenizer):
    messages = [
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : "古代の湖にて観測される甲殻類「ブリリア」について教えて下さい。markdown等は使用せず文章のみで回答してください"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : EsukaKnight}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : "古代の湖にて観測される甲殻類「ブリリア」の学名を考えてください。2単語で。見た瞬間に意味がわかるようなわかりやすいものは避けてください。学名のみを答えてください",}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : "Testaceobrachia propulsus"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : "いいですね。次はアンカラ洞窟にて観測される架空の生物「ザトン」について教えて下さい。markdown等は使用せず文章のみで回答してください",}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : Kyomuton}
            ]
        },{
            "role": "user",
            "content": [
                {"type" : "text", "text" : "アンカラ洞窟にて観測される架空の生物「ザトン」の学名を考えてください。2単語で。見た瞬間に意味がわかるようなわかりやすいものは避けてください。学名のみを答えてください",}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : "Spelaeoneura parietalis"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : f"素晴らしいですね。次は{target}について教えて下さい。markdown等は使用せず文章のみで回答してください",}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : description}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : f"{target}の学名を考えてください。2単語で。見た瞬間に意味がわかるようなわかりやすいものは避けてください。学名のみを答えてください",}
            ]
        },
        
    ]
    return generate_text(messages, model, tokenizer)


def generate_prompt(target, description, model, tokenizer):
    sample_prompt = """
    masterpiece, best quality, absurdres, intricate cave system, vast underground cavern, damp and misty atmosphere, rugged cave walls, uneven rocky surfaces, small flying insects swarming, faint bioluminescent glow, scattered moss patches, luminescent fungi, hanging roots, water droplets falling from stalactites, quiet and eerie ambiance, hidden crevices, ancient untouched cave, faint echoes, cold and humid air, mysterious and unexplored, tiny fast-moving creatures clinging to walls, large round reflective eyes, twitching limbs, blurry movement, sharp focus on one specimen, watchful gaze, poised to strike at passing insect, stark contrast of delicate organism and rough stone, discovery scene, flashlight beam illuminating creature, sense of rare encounter, scientific expedition vibe
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : f"アンカラ洞窟にて観測される架空の生物「ザトン」は以下のような生物です。\n\n{Kyomuton}\n\nそしてこの生物のイメージを描くためのプロンプトが以下のとおりです。\n\n{sample_prompt}\n\n。これにならって、以下のような{target}のイメージを描くためのプロンプトを英語で作成してください。\n\n{description}\n\n生物が大型の場合はその動物を中心に、小型の場合は生息地を中心とした絵を描くようにしてください。プロンプトのみを答え、解説等はしないでください。あなたの出力はそのままStable Diffusionに渡されます"}
            ]
        }
        
    ]
    return generate_text(messages, model, tokenizer)


def generate_description(target, model, tokenizer):
    messages = [
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : "古代の湖にて観測される甲殻類「ブリリア」について3から5文程度で教えて下さい。markdown等は使用せず文章のみで回答してください"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : EsukaKnight}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : "いいですね。次は深海にて観測される架空の生物「ミズモドキ」について3から5文程度で教えて下さい。markdown等は使用せず文章のみで回答してください"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : Mizumodoki}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : "いいですね。次はアンカラ洞窟にて観測される架空の生物「ザトン」について3から5文程度で教えて下さい。markdown等は使用せず文章のみで回答してください",}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text" : Kyomuton}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : f"いいですね。次は{target}について3から5行程度で教えて下さい。markdown等は使用せず文章のみで回答してください",}
            ]
        }
    ]
    return generate_text(messages, model, tokenizer)



def generate_text_(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens = 1024, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    
    output = output.cpu()
    result = tokenizer.decode(output[0])[len(text):].replace("<end_of_turn>", "")
    del inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return result.strip()

def generate_text(messages, model, tokenizer):
    out1 = generate_text_(messages, model, tokenizer)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type" : "text", "text": out1}
            ]
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type" : "text", "text" : f"今の回答を60点として、userの指示に沿った100点の回答をしてください。改善点等の解説はせず、回答のみを答えてください。長さを水増しするのではなく中身の質を上げてください。"}
            ]
        }
    )
    return generate_text_(messages, model, tokenizer)