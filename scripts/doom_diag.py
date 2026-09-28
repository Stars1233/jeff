"""Which button a Jeff server picks, by where the nearest monster is, on Doom states from the rule bot's play (seed 1234),
with the "situation" wording. Usage: python scripts/doom_diag.py URL"""
import collections, os, sys
import vizdoom as vzd
from jeff.games import common, doom
client = common.JeffClient(sys.argv[1])
g = vzd.DoomGame(); g.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg")); g.set_window_visible(False)
g.set_objects_info_enabled(True)
g.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])
g.set_seed(1234); g.init(); g.new_episode()
table = collections.defaultdict(collections.Counter)
for _ in range(150):
    if g.is_episode_finished(): break
    desc = doom.describe(g.get_state())
    if desc["monsters_nearest_first"]:
        b = desc["monsters_nearest_first"][0]["bearing_deg"]
        where = "straight ahead" if abs(b) <= 8 else "to the left" if b > 0 else "to the right"
        state, options = doom.situation(desc)
        pick = client.ask(state, {"action": {"type": "choice", "instructions": doom.INSTRUCTIONS, "criteria": options}})["action"]["choice"]
        table[where][pick] += 1
    g.make_action([1 if b == doom.rule_player(desc) else 0 for b in doom.BUTTONS], 4)
for where in ("straight ahead", "to the left", "to the right"):
    print(f"  monster {where:15}: {dict(table[where])}")
