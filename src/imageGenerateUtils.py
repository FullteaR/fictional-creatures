from compel import Compel, ReturnedEmbeddingsType
from janome.tokenizer import Tokenizer
from PIL import Image, ImageDraw
import random

def get_image(prompt, pipe):
    negative_prompt = "bad quality,worst quality,worst detail,sketch,censor,logo,alphabet,watermark,nsfw,1girl,human"
    compel = Compel(
        truncate_long_prompts=False,
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )  
    prompt_embed, prompt_pooled = compel(prompt)
    negative_embed, negative_pooled = compel(negative_prompt)
    [prompt_embed, negative_embed] = compel.pad_conditioning_tensors_to_same_length([prompt_embed, negative_embed])
    image = pipe(
        prompt_embeds=prompt_embed, 
        pooled_prompt_embeds=prompt_pooled, 
        negative_prompt_embeds=negative_embed, 
        negative_pooled_prompt_embeds=negative_pooled, 
        width=800,
        height=480,
        guidance_scale=6,
        num_inference_steps=50,
        max_sequence_length=2048
    ).images[0]
    return image


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