"""What a Jeff server's Pac-Man choices say (consequence wording), on the same states from the rule bot's play.
Usage: python scripts/pacman_diag.py URL"""
import collections
import sys

from jeff.games import common, pacman

FOOD = ("eats a pellet right away", "pellets a few steps along", "pellets further away", "nothing left to eat")

client = common.JeffClient(sys.argv[1])
danger, food = collections.Counter(), collections.Counter()
reverse = safe_food_skipped = n = 0
game = pacman.PacMan("outcomes")
for seed in (1234, 1235, 1236):
    game.reset(seed)
    for _ in range(150):
        if game.done:
            break
        state, question = game.observe()
        if len(question["criteria"]) > 1:
            pick = client.ask(state, {"move": question})["move"]["choice"]
            text = question["criteria"][pick]
            n += 1
            danger[text.split(",")[0]] += 1
            food[next((f for f in FOOD if f in text), "-")] += 1
            reverse += pick == "reverse"
            best = [k for k, v in question["criteria"].items() if v.startswith("safe") and "right away" in v]
            safe_food_skipped += bool(best) and pick not in best
        # follow the rule bot, so every model sees exactly the same states
        game.version, game.criteria = "rule", pacman.CRITERIA["rule"]
        rule_state, rule_question = game.observe()
        move = pacman.rule_player(rule_state, rule_question)
        game.version, game.criteria = "outcomes", None
        game.step(move)
print(f"decisions with a real choice: {n}")
print("danger of the chosen option:", {k: f"{v / n:.0%}" for k, v in danger.items()})
print("food of the chosen option:  ", {k: f"{v / n:.0%}" for k, v in food.items()})
print(f"chose reverse: {reverse / n:.0%} | a safe option that eats right away existed but was not chosen: {safe_food_skipped / n:.0%}")
