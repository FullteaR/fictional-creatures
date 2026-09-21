"""裏設定の抽選。構図・体型・種との噛み合わせと、テーブルの整合。"""
import textGenerateUtils as t

ROLLS = 2000
SCHEMA = {
    "COMPOSITIONS": ["directive", "draws_creature", "group_only", "label", "limb_mode",
                     "magnifies_body", "negative", "palette", "shows_group", "solo", "weight"],
    "BODY_PLANS": ["colonial", "en", "head", "ja", "kind", "label", "limb_cap", "limb_heavy",
                   "limbs", "negative", "weight"],
    "PALETTES": ["conspicuous", "dark", "en", "ja", "kinds", "label", "weight"],
    "SURFACES": ["en", "ja", "kinds", "label", "weight"],
    "SIZES": ["en", "ja", "label", "weight"],
    "DANGERS": ["label", "weight"],
    "POPULATIONS": ["group", "label", "weight"],
    "REGISTERS": ["instruction", "label", "samples", "weight"],
    "PARTS": ["needs", "text"],
}


def row(rows, label):
    return [entry for entry in rows if entry["label"] == label][0]


def rolled(species=None, times=ROLLS):
    return [t.pick_traits(species) for _ in range(times)]


def test_every_table_row_carries_exactly_the_keys_the_code_reads():
    for name, keys in SCHEMA.items():
        for entry in getattr(t, name):
            assert sorted(entry) == keys, name


def test_camouflage_never_gets_a_conspicuous_palette():
    seen = 0
    for traits in rolled():
        if traits["composition"]["label"] == "擬態と保護色":
            seen += 1
            assert not traits["palette"]["conspicuous"]
    assert seen


def test_the_night_plate_never_gets_a_near_black_palette():
    seen = 0
    for traits in rolled():
        if traits["composition"]["label"] == "夜間の観察":
            seen += 1
            assert not traits["palette"]["dark"]
    assert seen


def test_palette_and_surface_suit_the_body_plan():
    for traits in rolled():
        kind = traits["body_plan"]["kind"]
        for key in ("palette", "surface"):
            kinds = traits[key]["kinds"]
            assert not kinds or kind in kinds


def test_a_species_that_names_a_scale_only_rolls_that_scale():
    assert {traits["size"]["label"] for traits in rolled("微生物", 200)} == {"微小"}
    assert {traits["size"]["label"] for traits in rolled("巨大生物", 200)} <= {"数メートル", "巨大"}


def test_a_species_that_names_a_form_only_rolls_that_form():
    assert {traits["body_plan"]["kind"] for traits in rolled("鳥", 200)} == {"翼"}


def test_an_unlisted_species_may_roll_any_form():
    assert len({traits["body_plan"]["kind"] for traits in rolled("生物")}) == len(t.DEFAULT_KINDS)


def test_the_swarm_plate_is_dealt_only_to_a_swarm():
    for traits in rolled():
        if traits["composition"]["group_only"]:
            assert traits["population"]["group"]


def test_a_limb_heavy_plan_is_never_magnified():
    for traits in rolled():
        if traits["body_plan"]["limb_heavy"]:
            assert not traits["composition"]["magnifies_body"]


def test_filled_supplies_every_trait_the_calls_read():
    assert sorted(t.filled(None)) == ["body_plan", "composition", "palette", "population",
                                      "register", "size", "surface"]


def test_filled_keeps_what_it_is_given():
    plan = row(t.BODY_PLANS, "群体型")
    assert t.filled({"body_plan": plan})["body_plan"] is plan


def test_roll_falls_back_when_every_weight_is_zero():
    rows = [{"weight": 0, "label": "a"}, {"weight": 0, "label": "b"}]
    assert t._roll(rows) in rows


def test_roll_falls_back_when_the_filter_empties_the_pool():
    rows = [{"weight": 1, "label": "a"}, {"weight": 1, "label": "b"}]
    assert t._roll(rows, lambda entry: False) in rows


def test_roll_honours_the_filter():
    rows = [{"weight": 1, "label": "a"}, {"weight": 1, "label": "b"}]
    assert t._roll(rows, lambda entry: entry["label"] == "b")["label"] == "b"


def test_a_limbless_plan_is_given_the_limbless_negative():
    negative = t.extra_negative({"body_plan": row(t.BODY_PLANS, "植物体型")})
    assert "standing on legs" in negative
    assert negative.count("claws") == 1


def test_a_limbed_plan_is_not():
    assert "standing on legs" not in t.extra_negative({"body_plan": row(t.BODY_PLANS, "節足型")})


def test_a_colony_is_never_drawn_as_one_specimen():
    traits = t.filled({"body_plan": row(t.BODY_PLANS, "群体型")})
    assert not t.draws_solo(traits)


def test_a_lone_creature_on_a_whole_body_plate_is():
    assert t.draws_solo(t.filled(None))


def test_a_distant_swarm_has_no_foreground_specimen():
    body_plan = row(t.BODY_PLANS, "節足型")
    traits = t.filled({
        "body_plan": body_plan,
        "composition": t._composition(row(t.COMPOSITIONS, "群れの遠景"), body_plan),
        "population": row(t.POPULATIONS, "大量発生しており、生息地では群れに出くわす"),
    })
    assert t.draws_group(traits)
    assert not t.draws_foreground_group(traits)


def test_every_plan_has_a_body_part_that_can_be_magnified():
    for plan in t.BODY_PLANS:
        assert t._parts(plan)


def test_a_headless_plan_is_never_asked_for_a_close_up_of_its_head():
    for plan in t.BODY_PLANS:
        if not plan["head"]:
            assert "the head" not in t._parts(plan)
            assert "the mouthparts" not in t._parts(plan)


def test_a_limbless_plan_is_never_asked_for_a_close_up_of_a_limb():
    for plan in t.BODY_PLANS:
        if plan["limbs"] == "none":
            assert "one limb" not in t._parts(plan)


def test_every_register_sample_carries_a_body_plan():
    for register in t.REGISTERS:
        for phrase, _ in register["samples"]:
            assert t._SAMPLE_PLANS[phrase] in t._PLAN_LABELS
