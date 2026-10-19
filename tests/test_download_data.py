"""scripts/download_data.py without the network: unpacking a .tar.zst, putting the .h5 where the code
expects it, and checking that the dataset has every column the task's protocol reads."""

import importlib.util
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

zstandard = pytest.importorskip("zstandard")
pytest.importorskip("stable_worldmodel")
from gplan.torch.evaluation import TASKS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_data", REPO / "scripts" / "download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_h5(path, columns):
    """A minimal LeWM-format dataset: 2 episodes of 30 frames with the given extra columns."""
    n = 60
    with h5py.File(path, "w") as f:
        f["pixels"] = np.zeros((n, 4, 4, 3), dtype=np.uint8)
        f["action"] = np.zeros((n, 2), dtype=np.float32)
        f["ep_idx"], f["step_idx"] = np.repeat([0, 1], 30), np.tile(np.arange(30), 2)
        f["ep_len"], f["ep_offset"] = np.array([30, 30]), np.array([0, 30])
        for col in columns:
            f[col] = np.zeros((n, 3), dtype=np.float32)


def make_archive(path, members):
    """members: {name inside the archive: local file} -> a .tar.zst at `path`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, local in members.items():
            tar.add(local, arcname=name)
    path.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))


def test_required_columns_follow_the_reset_calls(dl):
    assert dl.required_columns(TASKS["reacher"]) >= {"pixels", "action", "qpos", "qvel"}
    cube = dl.required_columns(TASKS["cube"])
    assert {"qpos", "qvel", "privileged_block_0_pos", "privileged_block_0_quat"} <= cube
    assert 0 not in cube and "cube_id" not in cube  # set_target_pos(cube_id=0) is a constant, not a column
    assert {"proprio"} <= dl.required_columns(TASKS["tworoom"]) and {"state"} <= dl.required_columns(TASKS["pusht"])


def test_find_archive(dl):
    assert dl.find_archive([".gitattributes", "README.md", "reacher.tar.zst"]) == "reacher.tar.zst"
    with pytest.raises(SystemExit):
        dl.find_archive(["README.md"])


@pytest.mark.parametrize("arcname", ["reacher_random.h5", "some/folder/reacher_random.h5", "data/other_name.h5"])
def test_unpack_and_place_wherever_the_archive_keeps_the_h5(dl, tmp_path, arcname):
    write_h5(tmp_path / "src.h5", ["qpos", "qvel"])
    make_archive(tmp_path / "reacher.tar.zst", {arcname: tmp_path / "src.h5", "README.md": REPO / "README.md"})
    dl.unpack(tmp_path / "reacher.tar.zst", tmp_path / "staging")
    target = tmp_path / "home" / "dmc" / "reacher_random.h5"
    dl.place_h5(tmp_path / "staging", target)
    assert target.exists()
    assert "2 episodes, 60 frames, 2-d actions" in dl.check_dataset(target, TASKS["reacher"])


def test_ambiguous_archive_is_refused(dl, tmp_path):
    write_h5(tmp_path / "a.h5", [])
    make_archive(tmp_path / "x.tar.zst", {"a.h5": tmp_path / "a.h5", "b.h5": tmp_path / "a.h5"})
    dl.unpack(tmp_path / "x.tar.zst", tmp_path / "staging")
    with pytest.raises(SystemExit):
        dl.place_h5(tmp_path / "staging", tmp_path / "reacher_random.h5")


def test_check_names_the_missing_reset_columns(dl, tmp_path):
    write_h5(tmp_path / "cube.h5", ["qpos", "qvel", "privileged_block_0_pos"])
    with pytest.raises(SystemExit, match="privileged_block_0_quat"):
        dl.check_dataset(tmp_path / "cube.h5", TASKS["cube"])
