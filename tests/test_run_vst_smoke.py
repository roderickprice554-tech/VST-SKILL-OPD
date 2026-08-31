import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))


def load_smoke_module():
    module_path = Path(__file__).parents[1] / "audit/run_vst_smoke.py"
    assert module_path.is_file(), "smoke runner is not implemented"
    import run_vst_smoke

    return run_vst_smoke


def test_causal_chunks_never_expose_future_frames():
    smoke = load_smoke_module()

    chunks = smoke.causal_chunks(tuple(range(7)), chunk_size=3)
    assert chunks == ((0, 1, 2), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6))
    for step, visible in enumerate(chunks):
        assert max(visible) <= min((step + 1) * 3 - 1, 6)


def test_read_seek_row_uses_byte_offset(tmp_path):
    smoke = load_smoke_module()

    path = tmp_path / "sample_with_seeks.jsonl"
    rows = [
        json.dumps({"text": "视频"}, ensure_ascii=False) + "\n",
        json.dumps({"text": "second"}, ensure_ascii=False) + "\n",
    ]
    path.write_text("".join(rows), encoding="utf-8")
    offset = len(rows[0].encode("utf-8"))
    assert smoke.read_seek_row(path, offset) == {"text": "second"}


def test_chat_template_string_reports_decoded_video_turns():
    smoke = load_smoke_module()
    assert hasattr(smoke, "count_video_turns")
    rendered = (
        "<|im_start|>user<|vision_start|><|video_pad|><|vision_end|>first"
        "<|im_start|>assistant memory"
        "<|im_start|>user<|vision_start|><|video_pad|><|vision_end|>second"
    )
    assert smoke.count_video_turns(rendered) == 2
