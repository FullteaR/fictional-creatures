"""解説文を1文ずつ直す2つのパス。差し替えの可否と、丸ごと破棄する条件。"""
import textGenerateUtils as t

DESCRIPTION = (
    "洞窟に生息する生物。"
    "葉が完全に静止している日は、入洞が避けるのが慣わしとなっている。"
    "中心から広がる腕で虫を捕らえて生きている。"
    "堆積物の中の有機物を摂取する。"
    "やわらかい胴から伸ばした数本の触手で杉の皮にへばりつく。"
    "体長は五ミリ前後である。"
    "記録は十例に満たない。"
)
POLISH = t._polish_forbidden(DESCRIPTION, "テスト", "生物", "洞窟", t.filled(None))
PROOF = t._proof_forbidden("生物", "洞窟", t.filled({"danger": "人間には全く無害"}))
FOUR = "一文目。二文目である。三文目である。四文目である。"


def polished(original, fixed):
    return t._natural_sentence(fixed, original, POLISH)


def proofed(original, fixed):
    return t._revised_sentence(fixed, original, PROOF)


def keep(fixed, original, forbidden):
    return t._as_sentence(fixed)


def passes(reply, max_edits=1, description=FOUR):
    return t._sentence_pass(reply, t._sentences(description), keep, {}, max_edits)


def test_a_wrong_particle_is_fixed():
    assert polished("葉が完全に静止している日は、入洞が避けるのが慣わしとなっている。",
                    "葉が完全に静止している日は、入洞を避けるのが慣わしとなっている。")


def test_a_kanji_the_description_lacks_is_refused():
    assert not polished("堆積物の中の有機物を摂取する。", "堆積物の中の栄養分を摂取する。")


def test_an_edit_that_only_reaches_the_tail_is_refused():
    assert not polished("堆積物の中の有機物を摂取する。",
                        "堆積物の中の有機物を摂取している。")


def test_losing_more_than_six_characters_is_refused():
    assert not polished("やわらかい胴から伸ばした数本の触手で杉の皮にへばりつく。",
                        "伸ばした数本の触手で杉の皮にへばりつく。")


def test_a_rewrite_too_far_from_the_original_is_refused():
    assert not polished("堆積物の中の有機物を摂取する。", "中の堆積物から有機物を摂取している。")


def test_changing_a_numeral_is_refused():
    assert not polished("体長は五ミリ前後である。", "体長は十ミリ前後である。")


def test_losing_the_habitat_word_is_refused():
    assert not polished("洞窟に生息する生物。", "中に生息する生物。")


def test_an_unchanged_sentence_is_not_an_edit():
    assert not polished("体長は五ミリ前後である。", "体長は五ミリ前後である。")


def test_a_missing_full_stop_is_added():
    assert polished("堆積物の中の有機物を摂取する。",
                    "堆積物の中から有機物を摂取する").endswith("。")


def test_the_plate_pass_accepts_a_rewrite():
    assert proofed("洞窟に生息する生物。", "洞窟の岩肌に張り付いて生活する生物。")


def test_the_plate_pass_refuses_losing_the_habitat_word():
    assert not proofed("洞窟に生息する生物。", "岩肌に張り付いて生活する。")


def test_the_plate_pass_refuses_a_hidden_trait_label():
    assert not proofed("洞窟に生息する生物。", "洞窟に生息する、人間には全く無害な生物。")


def test_the_plate_pass_refuses_a_much_longer_sentence():
    assert not proofed("洞窟に生息する生物。",
                       "洞窟の奥深く、光の届かない岩肌の割れ目に沿って静かに生息している生物であり、"
                       "その姿を目にした記録はごく限られている。")


def test_the_opening_sentence_is_never_rewritten():
    assert all(entry["ok"] for entry in passes("1: NG 別の一文目。"))


def test_one_edit_is_applied():
    review = passes("2: NG 直した二文目である。")
    assert [entry["ok"] for entry in review] == [True, False, True, True]
    assert review[1]["text"] == "直した二文目である。"


def test_more_edits_than_allowed_void_the_round():
    review = passes("2: NG 直した二文目である。\n3: NG 直した三文目である。")
    assert all(entry["ok"] for entry in review)


def test_two_edits_survive_when_the_pass_allows_two():
    review = passes("2: NG 直した二文目である。\n3: NG 直した三文目である。", max_edits=2)
    assert [entry["ok"] for entry in review] == [True, False, False, True]


def test_a_fix_that_repeats_another_sentence_voids_the_round():
    assert all(entry["ok"] for entry in passes("2: NG 三文目である。"))


def test_a_line_that_does_not_parse_is_ignored():
    assert all(entry["ok"] for entry in passes("ここは判定の行ではありません"))


def test_a_verdict_outside_the_range_is_ignored():
    assert all(entry["ok"] for entry in passes("9: NG 存在しない文。"))


def test_apply_proof_splices_only_the_changed_sentence():
    review = passes("2: NG 直した二文目である。")
    assert t.apply_proof(FOUR, review) == "一文目。直した二文目である。三文目である。四文目である。"


def test_apply_proof_leaves_an_all_clear_description_alone():
    assert t.apply_proof(FOUR, passes("1: OK\n2: OK\n3: OK\n4: OK")) == FOUR
