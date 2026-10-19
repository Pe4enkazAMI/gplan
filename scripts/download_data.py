"""Download a task's dataset from the Hugging Face Hub into $STABLEWM_HOME and check it.

    python scripts/download_data.py --task reacher     # -> $STABLEWM_HOME/dmc/reacher_random.h5
    python scripts/download_data.py --task cube        # -> $STABLEWM_HOME/ogbench/cube_single_expert.h5
    python scripts/download_data.py --task cube --check-only     # only check an existing .h5

Each dataset repo of the LeWM collection holds one .tar.zst archive (Reacher 24 GB, Cube 46 GB). The
archive is downloaded to $STABLEWM_HOME/.downloads (resumable: rerun after an interruption), unpacked
into a staging folder, and the .h5 is moved to where the code expects it: <TASKS[task].dataset>.h5.
The archive is deleted afterwards unless --keep-archive (unpacking needs about the archive's size
again in free disk space).

The check opens the .h5 and verifies the columns the protocol needs: frames, actions, episode layout,
and the dataset columns the env is reset from (gplan.torch.evaluation.TASKS[task].resets).
"""

import argparse
import importlib.util
import shutil
import tarfile
from pathlib import Path

import h5py

from gplan.data import STABLEWM_HOME
from gplan.torch.evaluation import TASKS

DATASET_REPOS = {  # https://huggingface.co/collections/quentinll/lewm
    "tworoom": "quentinll/lewm-tworooms",
    "pusht": "quentinll/lewm-pusht",
    "reacher": "quentinll/lewm-reacher",
    "cube": "quentinll/lewm-cube",
}


def find_archive(files):
    """The one .tar.zst of a dataset repo's file list."""
    archives = [f for f in files if f.endswith(".tar.zst")]
    if len(archives) != 1:
        raise SystemExit(f"expected one .tar.zst in the dataset repo, found {archives} among {files}")
    return archives[0]


def unpack(archive, staging):
    """Stream-decompress a .tar.zst into `staging` (only regular files and folders inside it)."""
    import zstandard

    staging.mkdir(parents=True, exist_ok=True)
    with open(archive, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(staging, filter="data")


def place_h5(staging, target):
    """Move the dataset's .h5 from the unpacked archive to `target`. The archive may hold it at the
    expected relative path or anywhere else: the .h5 with the target's file name is used, or the only
    .h5 in the archive."""
    found = sorted(staging.rglob("*.h5"))
    named = [p for p in found if p.name == target.name]
    if len(named) == 1:
        source = named[0]
    elif len(found) == 1:
        source = found[0]
    else:
        raise SystemExit(f"cannot tell which .h5 of the archive is {target.name}: {found}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), target)
    return source


def required_columns(task):
    """Columns the protocol reads: frames, actions, the episode layout, the cached columns and the
    columns the reset calls take (a goal_<col> argument is <col> of the goal row)."""
    columns = {"pixels", "action", "step_idx", "ep_len", "ep_offset", *task.cache_keys}
    for reset in task.resets:
        for arg in reset["args"].values():
            if arg.get("in_dataset", True):
                columns.add(arg["value"].removeprefix("goal_"))
    return columns


def check_dataset(path, task):
    """Raise if the .h5 lacks a column the task needs; return a one-line summary otherwise."""
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        missing = sorted(required_columns(task) - keys)
        if not ({"ep_idx", "episode_idx"} & keys):
            missing.append("ep_idx (or episode_idx)")
        if missing:
            raise SystemExit(f"{path} lacks the column(s) {missing} that the evaluation protocol needs; has {sorted(keys)}")
        n_frames, n_episodes = f["action"].shape[0], f["ep_len"].shape[0]
        return (f"{path}: {n_episodes} episodes, {n_frames} frames, {f['action'].shape[1]}-d actions, "
                f"frames {tuple(f['pixels'].shape[1:])}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(DATASET_REPOS))
    p.add_argument("--check-only", action="store_true", help="do not download, only check the existing .h5")
    p.add_argument("--keep-archive", action="store_true", help="keep the downloaded .tar.zst")
    p.add_argument("--force", action="store_true", help="download again even if the .h5 exists")
    args = p.parse_args()

    task = TASKS[args.task]
    target = STABLEWM_HOME / f"{task.dataset}.h5"
    if args.check_only or (target.exists() and not args.force):
        if not target.exists():
            raise SystemExit(f"{target} does not exist; run without --check-only to download it")
        print(check_dataset(target, task))
        return

    missing = [m for m in ("huggingface_hub", "zstandard") if importlib.util.find_spec(m) is None]
    if missing:  # check before a download of tens of GB, not after
        raise SystemExit(f"missing {missing}: pip install -e '.[convert]'")
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = DATASET_REPOS[args.task]
    archive_name = find_archive(list_repo_files(repo, repo_type="dataset"))
    downloads = STABLEWM_HOME / ".downloads"
    print(f"downloading {repo}/{archive_name} to {downloads} (resumable)")
    archive = Path(hf_hub_download(repo, archive_name, repo_type="dataset", local_dir=downloads))
    free = shutil.disk_usage(STABLEWM_HOME).free
    if free < archive.stat().st_size:
        print(f"WARNING: {free / 1e9:.0f} GB free, unpacking needs about {archive.stat().st_size / 1e9:.0f} GB")

    staging = STABLEWM_HOME / f".unpack-{args.task}"
    print(f"unpacking into {staging}")
    unpack(archive, staging)
    source = place_h5(staging, target)
    print(f"moved {source.relative_to(staging)} -> {target}")
    leftovers = [p for p in staging.rglob("*") if p.is_file()]
    if leftovers:
        print(f"the archive also held {len(leftovers)} other file(s), left in {staging}: {leftovers[:5]}")
    else:
        shutil.rmtree(staging)
    if not args.keep_archive:
        archive.unlink()
    print(check_dataset(target, task))


if __name__ == "__main__":
    main()
