"""キャプションの改行。語の途中で折らないこと、禁則、幅に収まること。"""
import os

import pytest
from PIL import ImageFont

import imageGenerateUtils as ig

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDTH = 420
SENTENCES = (
    "尾根の岩肌を覆い尽くす無定形の群体生物。個体は米粒ほどの半透明な球体だが、"
    "尾根の岩肌ではなくその足元の地面に円状に集まって生息している。"
    "この地では「道しるべの苔」と呼ばれており、集団が濃く色づいた方向に歩くことで"
    "水源や安全な避難場所を見つけられると信じられている。",
    "笹の葉を主食とする有蹄類の草食動物。生息域では糞便と足跡が報告されており、"
    "活体は未だ確認されていない。頭胴長は約六十センチメートル、"
    "体重は十二キロから十五キロと推定される。",
    "地中に生息する生物。前後左右の区別がなく、中心から多数の腕が放射状に伸びている。"
    "これらの腕を用いて周囲の木々の根を掴み、その栄養分を摂取して生活している。",
)


@pytest.fixture(scope="module")
def font():
    return ImageFont.truetype(os.path.join(SRC, "ipagp.ttf"), 15)


def word_starts(text):
    starts, at = set(), 0
    for word in ig.getWords(text):
        starts.add(at)
        at += len(word)
    return starts


def test_compound_noun_is_one_word():
    assert ig.getWords("淡い黄褐色を呈する。") == ["淡い", "黄褐色を", "呈する。"]


def test_number_keeps_its_counter():
    assert ig.getWords("四本の脚で体を支える。")[0] == "四本の"


def test_prefix_keeps_its_number_and_unit():
    assert ig.getWords("体長は約十五センチメートルである。") == ["体長は", "約十五センチメートルである。"]


def test_suffix_stays_with_its_noun():
    assert "登山者が" in ig.getWords("今も登山者が足を止める。")


def test_a_dependent_noun_stays_with_the_verb_before_it():
    assert "静止していることが" in ig.getWords("静止していることが多い。")
    assert "移動するために" in ig.getWords("移動するために体を縮める。")


def test_an_inflection_stays_with_its_verb():
    assert "みられる。" in ig.getWords("餌をとるものとみられる。")


def test_suru_verb_is_not_split_from_its_noun():
    assert "固定し、" in ig.getWords("一か所に体を固定し、移動せずに生活している。")


def test_compound_verb_is_one_word():
    assert "這い回る。" in ig.getWords("岩肌を這い回る。")


def test_opening_bracket_joins_what_follows():
    assert "「道しるべの" in ig.getWords("この地では「道しるべの苔」と呼ばれる。")


def test_a_verb_starts_a_word():
    assert "摂取して" in ig.getWords("その栄養分を摂取して生活している。")


def test_every_break_falls_between_words(font):
    for text in SENTENCES:
        starts = word_starts(text)
        at = 0
        for line in ig.getLineBreak(text, font, WIDTH)[:-1]:
            at += len(line)
            assert at in starts


def test_lines_fit_the_given_width(font):
    for text in SENTENCES:
        for width in (200, 300, WIDTH):
            for line in ig.getLineBreak(text, font, width):
                assert ig.getTextWidth(line, font) <= width


def test_nothing_is_lost_or_added(font):
    for text in SENTENCES:
        for width in range(80, 460, 20):
            assert "".join(ig.getLineBreak(text, font, width)) == text


def test_no_line_starts_on_forbidden_punctuation(font):
    for text in SENTENCES:
        for width in range(80, 460, 20):
            for line in ig.getLineBreak(text, font, width)[1:]:
                assert line[0] not in ig.HEAD_FORBIDDEN


def test_no_line_ends_on_an_opening_bracket(font):
    text = "この地では「道しるべの苔」と呼ばれ、見かけると豊作の兆しとされている。"
    for width in range(60, 460, 6):
        for line in ig.getLineBreak(text, font, width):
            assert line[-1] not in ig.TAIL_FORBIDDEN


def test_a_word_too_long_for_the_line_is_cut(font):
    text = "無定形の群体生物。"
    lines = ig.getLineBreak(text, font, 40)
    assert len(lines) > 1
    assert "".join(lines) == text


def test_the_tokenizer_is_built_once():
    assert ig.tokenizer() is ig.tokenizer()
