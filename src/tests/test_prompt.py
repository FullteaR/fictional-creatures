"""作画指示の組み立て。コードで前置きする指示と、各呼び出しに載る裏設定。"""
import pytest

import textGenerateUtils as t


def row(rows, label):
    return [entry for entry in rows if entry["label"] == label][0]


def traits(composition="生物の全体図", plan="節足型", palette="墨", surface="なめらか",
           size="数十センチ"):
    body_plan = row(t.BODY_PLANS, plan)
    return t.filled({
        "danger": "人間には全く無害",
        "body_plan": body_plan,
        "palette": row(t.PALETTES, palette),
        "surface": row(t.SURFACES, surface),
        "size": row(t.SIZES, size),
        "composition": t._composition(row(t.COMPOSITIONS, composition), body_plan),
    })


@pytest.fixture
def sent(monkeypatch):
    seen = []

    def fake(messages, max_tokens=2048):
        seen.append(messages)
        return "a creature in its habitat"

    monkeypatch.setattr(t, "call_llm", fake)
    return seen


def last(sent):
    return sent[-1][-1]["content"]


def test_the_composition_and_the_body_plan_are_prepended_in_order():
    picked = traits()
    prompt = t._with_directives("a creature", picked["composition"], False,
                                picked["body_plan"])
    assert prompt.index(picked["composition"]["directive"]) == 0
    assert prompt.index(picked["body_plan"]["en"]) < prompt.index("a creature")


def test_a_plate_without_the_creature_is_not_given_a_body_plan():
    picked = traits(composition="生態の痕跡")
    prompt = t._with_directives("tracks in mud", picked["composition"], False,
                                picked["body_plan"])
    assert picked["body_plan"]["en"] not in prompt


def test_a_directive_the_model_already_wrote_is_not_repeated():
    picked = traits()
    directive = picked["composition"]["directive"]
    prompt = t._with_directives(f"{directive}, a creature", picked["composition"], False, None)
    assert prompt.count("side-on specimen view") == 1


def test_one_matching_clause_is_not_an_echo():
    assert not t._echoed("one long clause here, another long clause, a third long clause",
                         "one long clause here, something else entirely")


def test_two_matching_clauses_are():
    assert t._echoed("one long clause here, another long clause, a third long clause",
                     "another long clause, one long clause here")


def test_an_article_does_not_hide_an_echo():
    assert t._echoed("the organism preserved as a dried specimen",
                     "one organism preserved as a dried specimen")


def test_the_prompt_call_carries_the_palette_and_the_surface(sent):
    picked = traits()
    t.generate_prompt("洞窟の生物", "説明文。", "裏設定。", picked)
    assert picked["palette"]["en"] in last(sent)
    assert picked["surface"]["en"] in last(sent)


def test_a_close_plate_is_not_told_the_scale(sent):
    t.generate_prompt("洞窟の生物", "説明文。", "裏設定。", traits())
    assert "大きさは" not in last(sent)


def test_a_habitat_plate_is(sent):
    picked = traits(composition="生息地の風景")
    t.generate_prompt("洞窟の生物", "説明文。", "裏設定。", picked)
    assert picked["size"]["en"] in last(sent)


def test_a_traces_plate_is_told_the_creature_is_out_of_frame(sent):
    t.generate_prompt("洞窟の生物", "説明文。", "裏設定。", traits(composition="生態の痕跡"))
    assert "no creature visible" in last(sent)


def test_a_limbless_plan_is_never_asked_for_numerals(sent):
    t.generate_prompt("洞窟の生物", "説明文。", "裏設定。", traits(plan="植物体型"))
    assert "英語の数詞" not in last(sent)
    assert "no legs, no arms" in last(sent)


def test_the_profile_call_dictates_the_first_two_lines(sent):
    picked = traits()
    t.generate_profile("洞窟の生物", "説明文。", picked)
    assert picked["size"]["ja"] in last(sent)
    assert picked["palette"]["ja"] in last(sent)
    assert picked["surface"]["ja"] in last(sent)


def test_the_profile_call_lets_the_description_win(sent):
    t.generate_profile("洞窟の生物", "説明文。", traits())
    assert "説明文が大きさや色に触れている場合" in last(sent)


def test_the_profile_call_caps_the_limbs_at_the_plans_own_limit(sent):
    t.generate_profile("洞窟の生物", "説明文。", traits(plan="四肢型"))
    assert "多くても4本" in last(sent)


def test_no_description_sample_writes_a_measurement_in_kanji():
    import re

    kanji_measure = re.compile(
        r"[一二三四五六七八九十百千]+(?:ミリ|センチ|メートル|キロ|グラム|リットル|例|年代)")
    for register in t.REGISTERS:
        for _, text in register["samples"]:
            assert not kanji_measure.search(text)


def test_the_description_call_asks_for_arabic_numerals(sent):
    t.generate_description("洞窟の生物", traits())
    assert "算用数字" in last(sent)


def test_the_description_call_is_given_the_body_plan_in_every_sample(sent):
    picked = traits()
    t.generate_description("洞窟の生物", picked)
    for message in sent[-1]:
        if message["role"] == "user":
            assert "この生物の姿は" in message["content"]
