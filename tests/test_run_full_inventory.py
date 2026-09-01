import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))

from run_full_inventory import extract_video_records, percentile, write_progress


def test_extract_video_records_from_message_list():
    row = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "dataset/example.mp4",
                    "video_start": 2.0,
                    "video_end": 7.5,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text_steam", "qa_stream": [[2.0, 7.5, "q", "a"]]}
            ],
        },
    ]

    records, qa_count = extract_video_records(row)

    assert records == [
        {
            "video": "dataset/example.mp4",
            "video_start": 2.0,
            "video_end": 7.5,
            "duration": 5.5,
        }
    ]
    assert qa_count == 1


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [([], 0.5, None), ([1, 2, 3], 0.0, 1.0), ([1, 2, 3], 0.5, 2.0), ([1, 2, 3], 1.0, 3.0)],
)
def test_percentile(values, fraction, expected):
    assert percentile(values, fraction) == expected


def test_write_progress_is_machine_readable(tmp_path):
    write_progress(tmp_path, "sft", 2, 6, "reading annotations")

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "sft"
    assert progress["completed_units"] == 2
    assert progress["total_units"] == 6
    assert progress["percent"] == pytest.approx(33.3333333333)
    assert progress["message"] == "reading annotations"
