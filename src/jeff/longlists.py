"""Long-list choice questions built in code: 20 to 254 options, so the model learns to answer with every option code,
including the two-letter ones after Z (AA, AB, ...). Without them no training question had more than 19 options and the
released models never picked an option past the 26th (https://github.com/firelex/jeff/issues/1).

Each row asks the model to find the one option the state refers to, among many look-alikes: a destination, a person, a
product, an order number, a meeting slot, a team or a setting. All names are made up from syllables, so no outside
knowledge is needed and nothing overlaps a benchmark. Some options are near-misses of the answer (one syllable or digit
different). The trainer shuffles the options at every step, so the answer lands at every position. At most 254 options,
because the mixer's escape step can add one more ("None of these") and the model takes at most 255.

Real long lists come from intent datasets asked with every intent as an option: Amazon MASSIVE (60 intents, train
split, CC BY 4.0) and CLINC150 (150 intents, train split, CC BY 3.0; its out-of-scope messages are asked with a
"None of these" option as the answer). Their test splits are never used."""

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path

from jeff.types import Example
from jeff.voice import intent_description, massive

CLINC = ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "plus/train-00000-of-00001.parquet")
NONE = ("none_of_these", "None of these")

MIN_OPTIONS, MAX_LIST = 20, 254
SYLLABLES = ("ka", "lo", "mi", "ren", "sa", "tor", "vel", "an", "bri", "do", "fen", "gar", "hel", "ist", "jo", "kel", "lin",
             "mar", "nor", "ol", "pa", "quin", "ros", "sel", "tan", "ul", "var", "wen", "yor", "zan", "cha", "del", "eth",
             "fal", "gri", "hon", "ira", "jun", "kor", "lum")
COLOURS = ("red", "blue", "green", "black", "white", "grey", "yellow", "orange", "purple", "brown", "pink", "navy", "teal")
MATERIALS = ("ceramic", "steel", "glass", "oak", "bamboo", "cotton", "wool", "leather", "linen", "copper", "plastic")
ITEMS = ("mug", "bowl", "lamp", "chair", "towel", "scarf", "bottle", "notebook", "backpack", "plate", "vase", "blanket",
         "cushion", "tray", "jar")
TOPICS = ("billing", "refunds", "password resets", "shipping delays", "damaged parcels", "account closures", "invoices",
          "tax forms", "app crashes", "login errors", "subscription changes", "gift cards", "address changes",
          "data exports", "privacy requests", "card payments", "bank transfers", "warranty claims", "returns",
          "delivery slots", "loyalty points", "price matching", "bulk orders", "API keys", "server outages")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def word(rng: random.Random, parts: int) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(parts)).capitalize()


def near_miss(rng: random.Random, name: str) -> str:
    """The name with one syllable-sized change, so the list holds look-alikes of the answer."""
    cut = rng.randrange(1, max(2, len(name) - 1))
    return (name[:cut] + rng.choice(SYLLABLES) + name[cut + 2:]).capitalize()


def distinct(rng: random.Random, make: Callable[[], str], answer: str, count: int, near: Callable[[str], str] | None) -> list[str]:
    """`count` distinct options including `answer`, with up to three near-misses of it."""
    options = {answer}
    if near is not None:
        for _ in range(rng.randint(1, 3)):
            options.add(near(answer))
    attempts = 0
    while len(options) < count:
        options.add(make())
        attempts += 1
        if attempts > count * 50:
            raise ValueError(f"Could not make {count} distinct options")
    result = sorted(options)
    rng.shuffle(result)
    return result


def list_size(rng: random.Random) -> int:
    """Between 20 and 254 options, log-uniform so lists near 27-100 are common and the largest still appear."""
    return int(math.exp(rng.uniform(math.log(MIN_OPTIONS), math.log(MAX_LIST + 1))))


def destination(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    make = lambda: word(rng, rng.randint(2, 3))  # noqa: E731
    answer = make()
    names = distinct(rng, make, answer, n, lambda a: near_miss(rng, a))
    state = rng.choice([f"Book me a flight to {answer} next Friday.", f"I need two train tickets to {answer}, please.",
                        f"What's the weather going to be like in {answer} this weekend?",
                        f"Can you find a hotel in {answer} for three nights?"])
    return state, "Which destination does the user mean?", {name: None for name in names}, answer


def person(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    make = lambda: f"{word(rng, 2)} {word(rng, rng.randint(2, 3))}"  # noqa: E731
    answer = make()
    first, last = answer.split(" ", 1)
    names = distinct(rng, make, answer, n, lambda a: f"{first} {near_miss(rng, last)}")
    state = rng.choice([f"Forward this email to {answer}.", f"Schedule a call with {answer} tomorrow morning.",
                        f"Add {answer} to the project channel.", f"Who approved this? It says {answer} signed it off."])
    return state, "Which contact is meant?", {name: None for name in names}, answer


def product(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    make = lambda: f"{rng.choice(COLOURS)} {rng.choice(MATERIALS)} {rng.choice(ITEMS)}"  # noqa: E731
    answer = make()
    colour, material, item = answer.split(" ")
    near = lambda a: f"{rng.choice([c for c in COLOURS if c != colour])} {material} {item}"  # noqa: E731
    count = min(n, len(COLOURS) * len(MATERIALS) * len(ITEMS))
    products = distinct(rng, make, answer, count, near)
    keys = {f"sku-{rng.randrange(10000, 99999)}-{i}": text for i, text in enumerate(products)}
    state = rng.choice([f"Order two more of the {answer}, please.", f"The {answer} I bought arrived cracked.",
                        f"Is the {answer} back in stock?"])
    return state, "Which product is the customer talking about?", dict(keys), next(k for k, v in keys.items() if v == answer)


def order_number(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    make = lambda: f"#{rng.randrange(100000, 999999)}"  # noqa: E731
    answer = make()
    near = lambda a: a[:-2] + f"{rng.randrange(100):02d}"  # noqa: E731
    numbers = distinct(rng, make, answer, n, near)
    state = rng.choice([f"Where is my order {answer}? It was due yesterday.", f"Please cancel order {answer}.",
                        f"I want to return everything from order {answer[1:]}."])
    return state, "Which order does the message refer to?", {number: None for number in numbers}, answer


def meeting_slot(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    slots = [f"{day} {hour:02d}:{minute:02d}" for day in DAYS for hour in range(7, 21) for minute in (0, 30)]
    answer = rng.choice(slots)
    day, clock = answer.split(" ")
    hour, minute = int(clock[:2]), int(clock[3:])
    spoken = f"{(hour - 1) % 12 + 1}{':30' if minute else ''} {'pm' if hour >= 12 else 'am'}"
    chosen = [answer] + rng.sample([s for s in slots if s != answer], min(n, len(slots)) - 1)
    rng.shuffle(chosen)
    state = rng.choice([f"Move my dentist appointment to {day} at {spoken}.", f"Can we meet {day}, {spoken}?",
                        f"Book the meeting room for {day} at {spoken}."])
    return state, "Which time slot does the user want?", {slot: None for slot in chosen}, answer


def team(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    count = min(n, len(TOPICS) * 6)
    pairs = [(topic, region) for topic in TOPICS for region in ("Europe", "Asia", "Americas", "Africa", "Oceania", "the Middle East")]
    rng.shuffle(pairs)
    pairs = pairs[:count]
    topic, region = rng.choice(pairs)  # not pairs[0]: the key's number must not give the answer away
    teams = {f"team-{word(rng, 2).lower()}-{i}": f"Handles {t} for customers in {r}." for i, (t, r) in enumerate(pairs)}
    keys = list(teams)
    rng.shuffle(keys)
    answer = next(k for k, v in teams.items() if v == f"Handles {topic} for customers in {region}.")
    state = f"Ticket from a customer in {region}: my problem is about {topic}, and nobody has replied for a week."
    return state, "Which team should get this ticket?", {k: teams[k] for k in keys}, answer


def setting(rng: random.Random, n: int) -> tuple[str, str, dict[str, str | None], str]:
    values = [f"{10 + t / 10:.1f} °C" for t in range(n)]  # 10.0 °C upwards in steps of 0.1: look-alikes by design
    answer = rng.choice(values)
    state = rng.choice([f"Set the living room to {answer.split(' ')[0]} degrees.",
                        f"Make it {answer.split(' ')[0]} in here, please."])
    rng.shuffle(values)
    return state, "Which temperature should the thermostat be set to?", {value: None for value in values}, answer


TEMPLATES: dict[str, Callable[[random.Random, int], tuple[str, str, dict[str, str | None], str]]] = {
    "destination": destination, "person": person, "product": product, "order_number": order_number,
    "meeting_slot": meeting_slot, "team": team, "setting": setting}


def build(count: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    rows: list[Example] = []
    names = list(TEMPLATES)
    for index in range(count):
        template = names[index % len(names)]
        state, instructions, criteria, label = TEMPLATES[template](rng, list_size(rng))
        if label not in criteria or not MIN_OPTIONS <= len(criteria) <= MAX_LIST:
            raise ValueError(f"{template} made a bad row: {len(criteria)} options, answer listed: {label in criteria}")
        identifier = f"longlist-{seed}-{index:06d}"
        rows.append({"id": identifier, "suite": "longlists", "family": identifier, "state": state,
                     "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
                     "label": label, "target": label,
                     "source": {"dataset": "longlists", "template": template, "options": len(criteria),
                                "license": "own (code-built)"}})
    return rows


def intent_row(identifier: str, text: str, intents: dict[str, str], label: str, source: dict[str, str]) -> Example:
    return {"id": identifier, "suite": "longlists", "family": identifier, "state": text,
            "question": {"type": "choice", "instructions": "Which of these does the user want?", "criteria": dict(intents)},
            "label": label, "target": label, "source": {**source, "options": len(intents)}}


def massive_lists(rows: list[dict], count: int, rng: random.Random) -> list[Example]:
    """MASSIVE training utterances, each asked with all of MASSIVE's intents as options."""
    names = sorted({row["intent"] for row in rows})
    intents = {name: intent_description(name) for name in names}
    return [intent_row(f"longlist-massive-{row['id']}", row["utt"], intents, row["intent"],
                       {"dataset": "longlists-massive", "upstream_id": str(row["id"]), "license": "CC BY 4.0"})
            for row in rng.sample(rows, min(count, len(rows)))]


def clinc_lists(rows: list[tuple[str, str]], count: int, rng: random.Random) -> list[Example]:
    """CLINC150 training utterances (text, intent), each asked with all 150 in-scope intents as options. Out-of-scope
    messages get a "None of these" option as the answer; a quarter of in-scope ones also list it, as a wrong option."""
    names = sorted({intent for _, intent in rows if intent != "oos"})
    intents = {name: name.replace("_", " ") for name in names}
    with_none = {**intents, NONE[0]: NONE[1]}
    scoped = [(i, r) for i, r in enumerate(rows) if r[1] != "oos"]
    result = []
    for index, (text, intent) in rng.sample(scoped, min(count, len(scoped))):
        options = with_none if rng.random() < 0.25 else intents
        result.append(intent_row(f"longlist-clinc-{index}", text, options, intent,
                                 {"dataset": "longlists-clinc150", "upstream_id": str(index), "license": "CC BY 3.0"}))
    for index, (text, _) in ((i, r) for i, r in enumerate(rows) if r[1] == "oos"):
        result.append(intent_row(f"longlist-clinc-{index}", text, with_none, NONE[0],
                                 {"dataset": "longlists-clinc150", "upstream_id": str(index), "license": "CC BY 3.0"}))
    return result


def clinc_train() -> list[tuple[str, str]]:
    """CLINC150 "plus" training split as (text, intent name), from the pinned Hugging Face revision."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, revision, filename = CLINC
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    table = pq.read_table(path)
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    names = features["intent"]["names"]
    return [(text, names[label]) for text, label in zip(table.column("text").to_pylist(), table.column("intent").to_pylist(), strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--intent-rows", type=int, default=6000, help="Utterances taken from each intent dataset")
    parser.add_argument("--out", type=Path, default=Path("data/longlists"))
    args = parser.parse_args()
    rng = random.Random(args.seed + 1)
    rows = build(args.count, args.seed) + massive_lists(massive("train"), args.intent_rows, rng) \
        + clinc_lists(clinc_train(), args.intent_rows, rng)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "train.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]["dataset"]] = by_source.get(row["source"]["dataset"], 0) + 1
    print(json.dumps({"rows": len(rows), "by_source": by_source, "out": str(args.out / "train.jsonl")}))


if __name__ == "__main__":
    main()
