import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_provenance_manifest():
    manifest_path = REPO_ROOT / "manifests" / "opd-foundation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_commit"] == "26f31d36eb8bcc0b480b43eaaed1277b33a10488"
    assert manifest["python"] == "/home/bujunru/.conda/envs/vision-se/bin/python"
    assert manifest["assets_copied"] is False
    assert manifest["models"]
    assert manifest["datasets"]

    for entry in manifest["models"] + manifest["datasets"]:
        assert Path(entry["path"]).is_absolute()
        assert entry["access"] == "read-only"
        assert len(entry["manifest_sha256"]) == 64
        int(entry["manifest_sha256"], 16)

    for relative_path, expected_hash in manifest["tracked_file_sha256"].items():
        actual_hash = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
