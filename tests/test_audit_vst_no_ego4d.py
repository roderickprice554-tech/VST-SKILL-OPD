import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))

from audit_vst_no_ego4d import classify_video, path_prefix


def test_path_prefix_keeps_top_level_component():
    assert path_prefix("LLaVA-Video-178K/a/b.mp4") == "LLaVA-Video-178K/"
    assert path_prefix("Ego4D/full_scale/a.mp4") == "Ego4D/"


def test_ego4d_is_excluded_even_when_missing(tmp_path):
    assert classify_video("Ego4D/full_scale/a.mp4", tmp_path) == "excluded_ego4d"


def test_existing_non_ego4d_media_is_ready(tmp_path):
    path = tmp_path / "hdvila/a.mp4"
    path.parent.mkdir()
    path.write_bytes(b"x")
    assert classify_video("hdvila/a.mp4", tmp_path) == "ready"


def test_missing_non_ego4d_media_is_blocking(tmp_path):
    assert (
        classify_video("LLaVA-Video-178K/a.mp4", tmp_path)
        == "missing_non_ego4d"
    )
