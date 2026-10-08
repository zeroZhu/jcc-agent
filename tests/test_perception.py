from jcc_agent.models import Hero, Ruleset
from jcc_agent.perception import FrameAnalyzer, OCRReader, OCRToken


class Image:
    shape = (100, 100, 3)


class Reader:
    def __init__(self, tokens):
        self.tokens = tokens

    def read(self, image):
        return self.tokens


def rules():
    return Ruleset("test", "test", {"ahri": Hero("ahri", "阿狸", 1),
                                     "garen": Hero("garen", "盖伦", 1)}, {}, {})


def token(text, x=5, y=5, confidence=0.99):
    return OCRToken(text, confidence, (x, y, x + 4, y + 4))


def test_paddle_3_adapter_preserves_box_and_confidence():
    class Engine:
        def predict(self, image):
            return [{"res": {"rec_texts": ["阿狸"], "rec_scores": [0.98],
                             "rec_polys": [[[1, 2], [20, 2], [20, 8], [1, 8]]]}}]
    assert OCRReader(Engine()).read(None) == [OCRToken("阿狸", 0.98, (1, 2, 20, 8))]


def test_missing_text_is_unknown_not_empty_and_timestamp_not_a_phase_oracle():
    analyzer = FrameAnalyzer(Reader([]), rules(), {"regions": {"shop_0": [0, 0, 1, 1]}})
    for time in (10, 35, 140, 300):
        state = analyzer.analyze(Image(), timestamp=time)
        assert state.phase == "unknown"
        assert all(slot.status == "unknown" for slot in state.shop)
        assert all(slot.source == "ocr_blank" for slot in state.shop)
        assert state.detail_open is None
        assert state.detail.status == "unknown"
        assert state.viewed_player == "unknown"


def test_exact_name_match_low_confidence_and_ambiguous_names_rejected():
    layout = {"regions": {"shop_0": [0, 0, .2, .2]}}
    for tokens in ([token("阿狸", confidence=.7)], [token("阿里")],
                   [token("阿狸"), token("盖伦", y=10)]):
        slot = FrameAnalyzer(Reader(tokens), rules(), layout).analyze(Image()).shop[0]
        assert slot.status == "unknown"
        assert slot.source == "ocr_unresolved"
    state = FrameAnalyzer(Reader([token("阿狸")]), rules(), layout).analyze(Image(), 8, 9)
    assert state.shop[0].hero_id == "ahri"
    assert state.shop[0].star is None
    assert state.shop[0].observed_at == 8
    assert state.version == 9


def test_spatial_regions_prevent_same_name_in_detail_becoming_shop_card():
    layout = {"regions": {"shop_0": [0, .8, 1, 1], "detail": [0, 0, .5, .5]}}
    state = FrameAnalyzer(Reader([token("盖伦")]), rules(), layout).analyze(Image())
    assert state.detail.hero_id == "garen"
    assert state.detail_open is True
    assert state.shop[0].status == "unknown"


def test_numeric_ambiguity_is_not_silently_resolved():
    layout = {"regions": {"gold": [0, 0, .5, .5], "level": [.5, 0, 1, .5]}}
    tokens = [token("50"), token("20", y=15), token("Lv.6", x=60)]
    state = FrameAnalyzer(Reader(tokens), rules(), layout).analyze(Image())
    assert state.gold is None
    assert state.level == 6


def test_positive_player_identity_and_explicit_detail_close_evidence():
    layout = {"regions": {"self_name": [0, 0, .5, .2], "detail_closed": [0, .5, 1, 1]},
              "self_names": ["测试玩家"], "detail_closed_text": ["测试关闭标记"]}
    state = FrameAnalyzer(Reader([token("测试玩家"), token("测试关闭标记", y=60)]), rules(), layout).analyze(Image())
    assert state.viewed_player == "self"
    assert state.detail_open is False
    assert state.detail is None
    assert state.field_confidence["detail_open"] == .99


def test_phase_comes_from_text_and_pve_round_rule_not_video_time():
    layout = {"regions": {"phase": [0, 0, .5, .5], "round": [.5, 0, 1, .5]},
              "phase_markers": {"augment": ["选择强化"], "preparation": ["备战"]},
              "pve_rounds": ["1-3"]}
    state = FrameAnalyzer(Reader([token("备战"), token("1-3", x=60)]), rules(), layout).analyze(Image(), 400)
    assert state.phase == "pve"
    assert state.round == "1-3"


def test_star_icons_not_interpreted_as_reliable_numeric_stars():
    layout = {"regions": {"detail": [0, 0, 1, 1]}}
    state = FrameAnalyzer(Reader([token("阿狸"), token("★★", y=20)]), rules(), layout).analyze(Image())
    assert state.detail.star is None
    state = FrameAnalyzer(Reader([token("阿狸"), token("二星", y=20)]), rules(), layout).analyze(Image())
    assert state.detail.star == 2


def test_shop_price_is_observed_not_inferred_from_pool_tier():
    fixture = rules()
    fixture.heroes["ahri"].cost = 2
    layout = {"regions": {"shop_0": [0, 0, 1, 1]}}
    state = FrameAnalyzer(Reader([token("阿狸"), token("1", y=20)]), fixture, layout).analyze(Image())
    assert state.shop[0].hero_id == "ahri"
    assert state.shop[0].cost == 1
    assert fixture.heroes["ahri"].cost == 2
    state = FrameAnalyzer(Reader([token("阿狸")]), fixture, layout).analyze(Image())
    assert state.shop[0].cost is None


def test_shop_round_timer_prove_gameplay_but_not_preparation():
    layout = {"regions": {"shop_0": [0, 0, .2, .2], "shop_1": [.2, 0, .4, .2],
                           "shop_2": [.4, 0, .6, .2], "round": [0, .5, .5, 1],
                           "timer": [.5, .5, 1, 1]}}
    tokens = [token("阿狸"), token("盖伦", x=25), token("阿狸", x=45),
              token("2-1", y=60), token("23", x=60, y=60)]
    state = FrameAnalyzer(Reader(tokens), rules(), layout).analyze(Image())
    assert state.phase == "gameplay"
    assert "phase_substage_unknown" in state.warnings
    state = FrameAnalyzer(Reader(tokens[:-1]), rules(), layout).analyze(Image())
    assert state.phase == "unknown"


def test_refresh_text_in_region_covered_by_detail_confirms_shop_and_closed_panel():
    layout = {"regions": {"refresh_label": [0, 0, 1, 1]},
              "field_regions": {"detail_closed": ["refresh_label"]}, "detail_closed_text": ["刷新"]}
    state = FrameAnalyzer(Reader([token("刷新")]), rules(), layout).analyze(Image())
    assert state.detail_open is False
    assert state.shop_visible is True
    assert state.field_confidence["shop_visible"] == .99
    state = FrameAnalyzer(Reader([]), rules(), layout).analyze(Image())
    assert state.detail_open is None
    assert state.shop_visible is None
