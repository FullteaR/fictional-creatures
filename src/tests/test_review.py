"""図版とカードの照合。項目の組み立て、注記の消毒、プロンプトへの反映。"""
import textGenerateUtils as t

PROMPT = ("segmented arthropod body, hard jointed exoskeleton, cave wall, "
          "six thin twitching legs, three legs on each side, no other limbs")


def row(rows, label):
    return [entry for entry in rows if entry["label"] == label][0]


def traits(composition="生物の全体図", plan="節足型", group=False):
    body_plan = row(t.BODY_PLANS, plan)
    return t.filled({
        "body_plan": body_plan,
        "composition": t._composition(row(t.COMPOSITIONS, composition), body_plan),
        "population": row(t.POPULATIONS, "大量発生しており、生息地では群れに出くわす")
        if group else row(t.POPULATIONS, "生息地では普通に見られる"),
    })


def keys(**kwargs):
    return [item["key"] for item in t._review_items(PROMPT, traits(**kwargs))]


def test_limb_numerals_are_pulled_out_of_the_prompt():
    assert t.limb_count_phrases(PROMPT) == ["six thin twitching legs",
                                            "three legs on each side", "no other limbs"]


def test_a_whole_creature_plate_is_asked_about_body_and_limbs():
    assert keys() == ["body_plan", "limbs", "composition", "solo", "body", "habitat"]


def test_a_traces_plate_is_not_asked_about_the_body():
    assert keys(composition="生態の痕跡") == ["absent", "composition", "habitat"]


def test_a_distant_swarm_is_not_asked_about_limb_counts():
    assert "limbs" not in keys(composition="群れの遠景", group=True)


def test_a_swarm_is_asked_for_many_individuals():
    assert "group" in keys(composition="生息地の風景", group=True)


def test_a_limbless_plan_is_asked_for_no_limbs_without_a_fix():
    items = t._review_items("no legs", traits(plan="植物体型"))
    limbs = [item for item in items if item["key"] == "limbs"][0]
    assert limbs["fix"] == ""


def test_a_note_that_starts_on_a_negation_is_dropped():
    assert t._review_note("no exoskeleton") == ""


def test_a_note_longer_than_six_words_is_dropped():
    assert t._review_note("a very long phrase with far too many words") == ""


def test_japanese_is_stripped_from_a_note():
    assert t._review_note("大きすぎる, brown body") == "brown body"


def test_at_most_two_notes_survive():
    assert t._review_note("one thing, two thing, three thing") == "one thing, two thing"


def test_a_verdict_line_is_parsed_into_its_item():
    items = t._review_items(PROMPT, traits())
    review = t._parse_review("1: NG single large insect\n2: OK", items)
    assert review[0]["ok"] is False
    assert review[0]["note"] == "single large insect"
    assert review[1]["ok"] is True


def test_an_item_with_no_verdict_defaults_to_ok():
    items = t._review_items(PROMPT, traits())
    assert all(entry["ok"] for entry in t._parse_review("", items))


def test_the_fix_moves_to_the_head_and_is_not_repeated():
    review = [{"key": "body_plan", "ok": False, "fix": "segmented arthropod body",
               "note": "single large insect"}]
    prompt, notes = t.apply_review(PROMPT, review)
    assert prompt.startswith("segmented arthropod body, ")
    assert prompt.count("segmented arthropod body") == 1
    assert notes == "single large insect"


def test_a_note_the_prompt_already_asks_for_is_dropped():
    review = [{"key": "composition", "ok": False, "fix": "", "note": "cave wall"}]
    assert t.apply_review(PROMPT, review)[1] == ""


def test_an_item_that_passed_contributes_nothing():
    review = [{"key": "composition", "ok": True, "fix": "wide view of the habitat",
               "note": "single large insect"}]
    assert t.apply_review(PROMPT, review) == (PROMPT, "")


def test_review_payload_keeps_only_what_the_card_stores():
    items = t._review_items(PROMPT, traits())
    payload = t.review_payload(t._parse_review("1: NG single large insect", items))
    assert sorted(payload[0]) == ["key", "note", "ok"]
