import random
import os
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from PIL import ImageFont
import torch
import matplotlib.pyplot as plt

from MonsterNameGenerator import MarkovMonsterNameGenerator
from textGenerateUtils import generate_scientific_name, generate_prompt, generate_description
from imageGenerateUtils import get_image, add_caption

# --- config ---
stable_pretrained_model_link_or_path = "plantMilkModelSuite_flax.safetensors"
monster_name_filepath = "monsterNames.txt"

title_font = ImageFont.truetype("ipagp.ttf", 27)
paragraph_font = ImageFont.truetype("ipagp.ttf", 15)
caption_font = ImageFont.truetype("ipagp.ttf", 12)

# --- SD pipeline ---
pipe = StableDiffusionXLPipeline.from_single_file(
    pretrained_model_link_or_path=stable_pretrained_model_link_or_path,
    torch_dtype=torch.float16
).to(device="cuda")

# --- name generator ---
name_generator = MarkovMonsterNameGenerator(n=2)
name_generator.train_from_file(monster_name_filepath)

fields = [
    "杉林", "古代林", "畑", "草むら", "花畑", "密林", "水没林", "ジャングル", "峠", "山の麓", "樹海", "竹林", "森", "霧の森", "熱帯雨林", "サバンナ", "桜並木", "果樹園",
    "洞窟", "鍾乳洞", "谷底", "岩石地帯", "鉱山", "荒野", "岩の中",
    "雪原", "凍土", "氷河",
    "旧市街地", "化学工場跡地", "都市の下水道", "古城", "都市部", "廃工場", "地下鉄廃線", "空中都市",
    "大砂漠", "オアシス",
    "海", "深海", "浅瀬", "砂浜", "汽水域", "川底", "孤島", "海底遺跡", "湖", "潮溜まり", "地下水路", "滝", "沈没船", "サンゴ礁",
    "成層圏", "惑星中心部", "溶岩地帯",
    "モンスターの体内"
]
spicies = [
    "生物", "鳥", "虫", "植物", "花", "草", "木", "キノコ", "魚", "爬虫類", "哺乳類", "両生類",
    "巨大生物", "小型生物", "草食動物", "肉食動物", "寄生生物", "絶滅危惧種", "甲殻類", "貝",
    "群生生物", "原始生物", "人工生命", "分類不明の生物"
]

# --- output dir ---
os.makedirs("endemic", exist_ok=True)

# --- main loop ---
finalImages = []
for j in tqdm(range(25)):
    name = name_generator.generate()
    field = random.choice(fields)
    if field == "モンスターの体内":
        field = name_generator.generate() + "の体内"
    spicy = random.choice(spicies)
    if spicy in ("貝", "草", "鳥", "魚") and random.randint(0, 1) == 1:
        name = name + spicy
    target = "{0}にて観測される架空の{1}「{2}」".format(field, spicy, name)
    print(target)

    description = generate_description(target)
    prompt = generate_prompt(target, description, pipe=pipe)
    scientific_name = generate_scientific_name(target, description)
    background = get_image(prompt.strip(), pipe)

    finalImage = add_caption(name, description, scientific_name, background, title_font, paragraph_font, caption_font)
    finalImage.save("endemic/{0}-{1}.png".format(j, name))
    finalImages.append(finalImage)

# --- summary plot ---
fig, axes = plt.subplots(5, 5, figsize=(40, 24))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
for ax, img in zip(axes.flatten(), finalImages):
    ax.imshow(img)
    ax.axis("off")
plt.savefig("endemic/summary.png", bbox_inches="tight")
plt.show()
