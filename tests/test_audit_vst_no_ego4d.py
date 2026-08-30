import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))

import audit_vst_no_ego4d as audit
from audit_vst_no_ego4d import classify_video, path_prefix


def test_path_prefix_keeps_top_level_component():
    assert path_prefix("LLaVA-Video-178K/a/b.mp4") == "LLaVA-Video-178K/"
    assert path_prefix("Ego4D/full_scale/a.mp4") == "Ego4D/"


def test_ego4d_is_excluded_even_when_missing(tmp_path):
    assert classify_video("Ego4D/full_scale/a.mp4", tmp_path) == "excluded_ego4d"


def test_nested_ego4d_path_is_excluded_case_insensitively(tmp_path):
    video = "LLaVA-Video-178K/source/EgO4D/a.mp4"
    assert classify_video(video, tmp_path) == "excluded_ego4d"


def test_ego4d_substring_is_not_treated_as_a_path_segment(tmp_path):
    video = "LLaVA-Video-178K/source/notego4d/a.mp4"
    assert classify_video(video, tmp_path) == "missing_non_ego4d"


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


def test_seek_index_uses_utf8_byte_offsets(tmp_path):
    jsonl = tmp_path / "sample_with_seeks.jsonl"
    rows = [
        json.dumps({"text": "ascii"}, ensure_ascii=False) + "\n",
        json.dumps({"text": "视频"}, ensure_ascii=False) + "\n",
        json.dumps({"text": "tail"}, ensure_ascii=False) + "\n",
    ]
    jsonl.write_text("".join(rows), encoding="utf-8")
    assert hasattr(audit, "write_seek_index")
    seek_path = audit.write_seek_index(jsonl)
    assert seek_path.name == "sample_seeks.jsonl"
    assert json.loads(seek_path.read_text(encoding="utf-8")) == [
        0,
        len(rows[0].encode("utf-8")),
        len((rows[0] + rows[1]).encode("utf-8")),
    ]


def test_non_ego4d_missing_prefixes_block_audit_completion():
    assert hasattr(audit, "blocking_missing_prefixes")
    assert audit.blocking_missing_prefixes(
        {"missing_prefixes": ["Ego4D/", "LLaVA-Video-178K/"]}
    ) == ("LLaVA-Video-178K/",)
    assert audit.blocking_missing_prefixes(
        {"missing_prefixes": ["Ego4D/"]}
    ) == ()
