import json
import tempfile
import unittest
from pathlib import Path

from jcc_agent.rules import load_compositions, load_rules, match_hero

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "inkborn_fables.json"


class RulesTests(unittest.TestCase):
    def test_roster_shared_lovers_and_templates(self):
        rules = load_rules(CONFIG)
        self.assertEqual(59, len(rules.heroes))
        self.assertEqual("xayah_rakan", match_hero("霞", rules))
        self.assertEqual("xayah_rakan", match_hero("洛", rules))
        self.assertEqual(5, len(load_compositions(CONFIG)))
        self.assertNotEqual("jcc_verified", rules.provenance)

    def test_matching_only_complete_names_or_reviewed_aliases(self):
        rules = load_rules(CONFIG)
        for text in ("阿狸", " 阿 狸 \n", "Ａｈｒｉ"):
            self.assertEqual("ahri", match_hero(text, rules))
        self.assertEqual("khazix", match_hero("Kha’Zix", rules))
        for text in ("阿理", "不是阿狸", "阿狸盖伦", "", "凯尔", "阿"):
            self.assertIsNone(match_hero(text, rules))

    def test_reject_invalid_odds_ambiguous_alias_or_template(self):
        for mutate in (
            lambda data: data["shop_odds"].update({"8": [0.1] * 5}),
            lambda data: data["heroes"][1]["aliases"].append("阿狸"),
            lambda data: data["heroes"][1].update({"cost": True}),
            lambda data: data["compositions"][0]["targets"].update({"unknown": 3}),
            lambda data: data.update({"pool_by_cost": []}),
            lambda data: data["shop_odds"].update({"01": [1, 0, 0, 0, 0]}),
        ):
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
            mutate(data)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_compositions(path)


if __name__ == "__main__":
    unittest.main()
