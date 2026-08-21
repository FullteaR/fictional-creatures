from compel import CompelForSDXL
from janome.tokenizer import Tokenizer
from PIL import Image, ImageDraw
import random
import torch

def get_image(prompt, pipe):
    negative_prompt = "lowres,early,monochrome,greyscale,worst quality,bad_quality,normal quality,lowres,anatomical nonsense,bad anatomy,anatomical nonsense,watermark,simple background,transparent,bad_feet,bad_hands,logo,text,bad_anatomy,signature,face backlighting,(worst quality, bad quality:1.2),jpeg artifacts,censored,extra digit,ugly,deformed anatomy,bad proportions"

    compel = CompelForSDXL(pipe)


    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        cond = compel(prompt, negative_prompt=negative_prompt)

    latents = pipe(
        prompt_embeds=cond.embeds,
        pooled_prompt_embeds=cond.pooled_embeds,
        negative_prompt_embeds=cond.negative_embeds,
        negative_pooled_prompt_embeds=cond.negative_pooled_embeds,
        width=800,
        height=480,
        guidance_scale=6,
        num_inference_steps=50,
        output_type="latent",
    ).images

    # VAE decode も float16 では NaN → float32 で実行
    pipe.vae.to(torch.float32)
    with torch.no_grad():
        decoded = pipe.vae.decode(
            latents.to(torch.float32) / pipe.vae.config.scaling_factor
        ).sample
    pixels = (decoded / 2 + 0.5).clamp(0, 1)
    image_np = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(image_np)


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

    x = random.choice([random.randint(20, 70), random.randint(im_w-max_line_width-70, im_w-max_line_width-20)])
    y = random.choice([random.randint(30, 80), random.randint(im_h-total_height-60, im_h-total_height-10)])
    padding = 10
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
