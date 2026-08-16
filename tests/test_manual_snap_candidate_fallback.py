from types import SimpleNamespace

from src.assembleur_tk import TriangleViewerManual


def _select(candidates, accepted_names):
    attempts = []

    def build_intent(candidate):
        name = candidate[1]
        attempts.append(name)
        return name

    def preview_intent(intent):
        return SimpleNamespace(accepted=intent in accepted_names)

    selected, intent, preview = (
        TriangleViewerManual._select_first_accepted_manual_attachment_candidate(
            candidates,
            build_intent=build_intent,
            preview_intent=preview_intent,
        )
    )
    return selected, intent, preview, attempts


def test_manual_snap_uses_next_candidate_when_best_preview_is_rejected():
    c1 = (0.1, "C1", "target-1")
    c2 = (0.2, "C2", "target-2")

    selected, intent, preview, attempts = _select([c2, c1], {"C2"})

    assert attempts == ["C1", "C2"]
    assert selected == c2
    assert intent == "C2"
    assert preview.accepted is True


def test_manual_snap_keeps_no_candidate_when_all_previews_are_rejected():
    c1 = (0.1, "C1", "target-1")
    c2 = (0.2, "C2", "target-2")

    selected, intent, preview, attempts = _select([c1, c2], set())

    assert attempts == ["C1", "C2"]
    assert selected is None
    assert intent is None
    assert preview is None


def test_manual_snap_stops_after_first_accepted_candidate():
    c1 = (0.1, "C1", "target-1")
    c2 = (0.2, "C2", "target-2")

    selected, intent, preview, attempts = _select([c1, c2], {"C1", "C2"})

    assert attempts == ["C1"]
    assert selected == c1
    assert intent == "C1"
    assert preview.accepted is True
