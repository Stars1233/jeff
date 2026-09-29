import random
from collections import Counter

from jeff.longlists import MAX_LIST, MIN_OPTIONS, NONE, TEMPLATES, build, clinc_lists, massive_lists


def test_every_row_has_one_listed_answer_and_a_long_list() -> None:
    rows = build(700, seed=1)
    assert Counter(row["source"]["template"] for row in rows) == {name: 100 for name in TEMPLATES}
    for row in rows:
        criteria = row["question"]["criteria"]
        assert MIN_OPTIONS <= len(criteria) <= MAX_LIST
        assert row["label"] == row["target"] and row["label"] in criteria
        assert len(set(criteria)) == len(criteria)


def test_lists_reach_past_z_and_up_to_the_limit() -> None:
    sizes = [len(row["question"]["criteria"]) for row in build(2000, seed=2)]
    assert sum(size > 26 for size in sizes) > 0.7 * len(sizes)
    assert max(sizes) > 200


def test_the_state_names_the_answer() -> None:
    for row in build(210, seed=3):
        template, state, label = row["source"]["template"], row["state"], row["label"]
        text = row["question"]["criteria"][label]
        if template in ("destination", "person"):
            assert label in state
        elif template == "product":
            assert text in state
        elif template == "order_number":
            assert label[1:] in state
        elif template == "setting":
            assert f"{label.split(' ')[0]} " in state
        elif template == "meeting_slot":
            assert label.split(" ")[0] in state


def test_building_is_reproducible() -> None:
    assert build(50, seed=4) == build(50, seed=4)


def test_intent_lists_offer_every_intent() -> None:
    massive = [{"id": i, "utt": f"wake me at {i}", "intent": intent}
               for i, intent in enumerate(["alarm_set", "alarm_query", "iot_hue_lightoff"] * 10)]
    rows = massive_lists(massive, 5, random.Random(0))
    assert len(rows) == 5 and all(len(row["question"]["criteria"]) == 3 for row in rows)
    assert all(row["label"] in row["question"]["criteria"] for row in rows)
    clinc = [("what's my balance", "balance"), ("book a table", "restaurant_reservation"), ("sing to me", "oos")]
    rows = clinc_lists(clinc, 2, random.Random(0))
    oos = [row for row in rows if row["label"] == NONE[0]]
    assert len(oos) == 1 and NONE[0] in oos[0]["question"]["criteria"]
    assert all("oos" not in row["question"]["criteria"] for row in rows)


def test_the_answer_key_does_not_give_the_answer_away() -> None:
    teams = [row for row in build(1400, seed=7) if row["source"]["template"] == "team"]
    assert len({row["label"].rsplit("-", 1)[1] for row in teams}) > 20
