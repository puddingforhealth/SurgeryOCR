#!/usr/bin/env python3
"""Move surgery videos and analysis results between this checkout and a GCS bucket.

The same tool runs on both ends:

    # workstation (videos live here)
    python tools/cloud_videos.py push data/0816 data/1131009何素珠od.MP4
    python tools/cloud_videos.py list

    # cloud machine (fresh clone, no videos yet)
    python tools/cloud_videos.py doctor
    python tools/cloud_videos.py pull --all --run          # download, then run_pipeline.py
    python tools/cloud_videos.py push-results --all        # send data/<video>/ outputs back

    # workstation again
    python tools/cloud_videos.py pull-results --all

Object keys mirror the path relative to ``data/`` so ``pull`` recreates the same
layout the golden-regression manifest (``tools/golden_videos.json``) expects.
Transfers go through ``gcloud storage``, which handles resumable multi-GB uploads
and needs no extra Python packages; authenticate once with ``gcloud auth login``
(``--no-browser`` on a headless machine).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config" / "cloud.json"
VIDEO_SUFFIXES = {".mp4"}
RESULT_GLOBS = ("*.json", "*.jsonl")


# --------------------------------------------------------------------------- config

def load_config(bucket_override: Optional[str]) -> Dict[str, str]:
    cfg: Dict[str, str] = {
        "bucket": "",
        "videos_prefix": "videos",
        "results_prefix": "results",
        "data_dir": "data",
    }
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    bucket = bucket_override or os.environ.get("SURGERYOCR_GCS_BUCKET") or cfg["bucket"]
    if not bucket:
        sys.exit("no bucket configured: set it in config/cloud.json, SURGERYOCR_GCS_BUCKET, or --bucket")
    if not bucket.startswith("gs://"):
        bucket = "gs://" + bucket
    cfg["bucket"] = bucket.rstrip("/")
    return cfg


def gcloud_bin() -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        sys.exit("gcloud not found on PATH. Run scripts/setup.sh (Linux) or install the Google Cloud SDK.")
    return exe


def gcloud(args: Sequence[str], capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [gcloud_bin(), *args]
    return subprocess.run(
        cmd, check=False, text=True, encoding="utf-8", errors="replace",
        capture_output=capture,
    )


# --------------------------------------------------------------------------- listing

def list_objects(bucket: str, prefix: str) -> List[Dict[str, str]]:
    """Return [{name, size}] for every object under ``prefix/``."""
    proc = gcloud(
        ["storage", "objects", "list", f"{bucket}/{prefix}/**", "--format", "json(name,size)"],
        capture=True,
    )
    if proc.returncode != 0:
        if "matched no objects" in (proc.stderr or ""):
            return []
        sys.exit(f"listing {bucket}/{prefix} failed:\n{proc.stderr.strip()}")
    rows = json.loads(proc.stdout or "[]")
    out = [{"name": r["name"], "size": int(r.get("size", 0))} for r in rows if r["name"].startswith(prefix + "/")]
    return sorted(out, key=lambda r: r["name"])


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GB"


def rel_key(obj_name: str, prefix: str) -> str:
    return obj_name[len(prefix) + 1:]


def select(keys: Iterable[str], patterns: Sequence[str], select_all: bool) -> List[str]:
    """Pick keys by exact name, basename, stem, substring, or glob (matched against the key and its basename)."""
    keys = list(keys)
    if select_all:
        return keys
    chosen: List[str] = []
    for key in keys:
        base = PurePosixPath(key).name
        for pat in patterns:
            pat_posix = pat.replace("\\", "/")
            if pat_posix.startswith("data/"):
                pat_posix = pat_posix[len("data/"):]
            if pat_posix in key or fnmatch.fnmatch(key, pat_posix) or fnmatch.fnmatch(base, pat_posix):
                chosen.append(key)
                break
    return chosen


# --------------------------------------------------------------------------- transfers

def copy(src: str, dst: str, no_clobber: bool) -> bool:
    args = ["storage", "cp"]
    if no_clobber:
        args.append("--no-clobber")
    args += [src, dst]
    proc = gcloud(args)
    return proc.returncode == 0


def local_videos(paths: Sequence[str], data_dir: Path) -> List[Path]:
    found: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            found += sorted(x for x in path.iterdir() if x.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            found.append(path)
        else:
            print(f"skip (not found): {p}")
    return found


def key_for(path: Path, data_dir: Path) -> str:
    """Key relative to data/; a video outside data/ is keyed by its basename."""
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- commands

def cmd_list(cfg: Dict[str, str], args: argparse.Namespace) -> int:
    for prefix, label in ((cfg["videos_prefix"], "videos"), (cfg["results_prefix"], "results")):
        rows = list_objects(cfg["bucket"], prefix)
        total = sum(r["size"] for r in rows)
        print(f"{label}: {len(rows)} objects, {human(total)}  ({cfg['bucket']}/{prefix}/)")
        if label == "results":
            # results are many small files; show one line per video directory
            dirs: Dict[str, int] = {}
            for r in rows:
                top = result_dir_key(rel_key(r["name"], prefix))
                dirs[top] = dirs.get(top, 0) + 1
            for d, n in sorted(dirs.items()):
                print(f"  {d}/  ({n} files)")
        else:
            for r in rows:
                print(f"  {human(r['size']):>10}  {rel_key(r['name'], prefix)}")
    return 0


def cmd_push(cfg: Dict[str, str], args: argparse.Namespace) -> int:
    data_dir = REPO / cfg["data_dir"]
    videos = local_videos(args.paths, data_dir)
    if not videos:
        print("nothing to push")
        return 1
    existing = {rel_key(r["name"], cfg["videos_prefix"]): r["size"] for r in list_objects(cfg["bucket"], cfg["videos_prefix"])}
    failed = 0
    for i, v in enumerate(videos, 1):
        key = key_for(v, data_dir)
        size = v.stat().st_size
        print(f"[{i}/{len(videos)}] {key}  ({human(size)})")
        if not args.force and existing.get(key) == size:
            print("   already in bucket with the same size, skipped")
            continue
        dst = f"{cfg['bucket']}/{cfg['videos_prefix']}/{key}"
        if not copy(str(v), dst, no_clobber=not args.force):
            print("   upload failed")
            failed += 1
    print(f"done, {len(videos) - failed}/{len(videos)} uploaded")
    return 1 if failed else 0


def run_pipeline(video: Path, mode: str) -> bool:
    cmd = [sys.executable, str(REPO / "run_pipeline.py"), "--video", str(video), "--mode", mode]
    print("   $ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, check=False).returncode == 0


def cmd_pull(cfg: Dict[str, str], args: argparse.Namespace) -> int:
    data_dir = REPO / cfg["data_dir"]
    rows = list_objects(cfg["bucket"], cfg["videos_prefix"])
    keys = select((rel_key(r["name"], cfg["videos_prefix"]) for r in rows), args.names, args.all)
    sizes = {rel_key(r["name"], cfg["videos_prefix"]): r["size"] for r in rows}
    if not keys:
        print("no matching videos in the bucket (try `list`)")
        return 1
    failed: List[str] = []
    pulled: List[Path] = []
    for i, key in enumerate(keys, 1):
        dest = data_dir / Path(key)
        print(f"[{i}/{len(keys)}] {key}  ({human(sizes[key])})")
        if dest.exists() and dest.stat().st_size == sizes[key] and not args.force:
            print("   already present with the same size, skipped download")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not copy(f"{cfg['bucket']}/{cfg['videos_prefix']}/{key}", str(dest), no_clobber=False):
                print("   download failed")
                failed.append(key)
                continue
        pulled.append(dest)
        if args.run:
            if not run_pipeline(dest, args.mode):
                print("   pipeline failed")
                failed.append(key)
                continue
            if args.upload_results:
                push_results_for(cfg, dest, data_dir, force=True)
    print(f"done, {len(keys) - len(failed)}/{len(keys)} ok" + (f", failed: {failed}" if failed else ""))
    return 1 if failed else 0


def result_dir_key(result_rel_key: str) -> str:
    """'0816/<stem>/stage_analysis.json' -> '0816/<stem>' (the per-video directory in the bucket)."""
    return result_rel_key.rsplit("/", 1)[0]


def result_dir_for(video: Path, data_dir: Path) -> Path:
    # run_pipeline.py writes to <output_dir>/<video stem>/ with output_dir = data/
    return data_dir / video.stem


def push_results_for(cfg: Dict[str, str], video: Path, data_dir: Path, force: bool) -> bool:
    rdir = result_dir_for(video, data_dir)
    files = [f for g in RESULT_GLOBS for f in rdir.glob(g)]
    if not files:
        print(f"   no results in {rdir}")
        return False
    dst = f"{cfg['bucket']}/{cfg['results_prefix']}/{key_for(video, data_dir).rsplit('.', 1)[0]}/"
    ok = True
    for f in sorted(files):
        ok &= copy(str(f), dst, no_clobber=not force)
    print(f"   {len(files)} result files -> {dst}")
    return ok


def cmd_push_results(cfg: Dict[str, str], args: argparse.Namespace) -> int:
    data_dir = REPO / cfg["data_dir"]
    if args.all:
        videos = [p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES]
    else:
        videos = local_videos(args.names, data_dir)
    videos = [v for v in videos if result_dir_for(v, data_dir).is_dir()]
    if not videos:
        print("no result directories found")
        return 1
    ok = True
    for v in videos:
        print(key_for(v, data_dir))
        ok &= push_results_for(cfg, v, data_dir, force=args.force)
    return 0 if ok else 1


def cmd_pull_results(cfg: Dict[str, str], args: argparse.Namespace) -> int:
    data_dir = REPO / cfg["data_dir"]
    rows = list_objects(cfg["bucket"], cfg["results_prefix"])
    dirs = sorted({result_dir_key(rel_key(r["name"], cfg["results_prefix"])) for r in rows})
    chosen = select(dirs, args.names, args.all)
    if not chosen:
        print("no matching result directories in the bucket (try `list`)")
        return 1
    ok = True
    for d in chosen:
        # results are keyed by the video key without extension; local output dir is flat under data/
        dest = data_dir / PurePosixPath(d).name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{d}/ -> {dest}")
        ok &= copy(f"{cfg['bucket']}/{cfg['results_prefix']}/{d}/*", str(dest), no_clobber=False)
    return 0 if ok else 1


def cmd_doctor(cfg: Dict[str, str], args: argparse.Namespace) -> int:
    problems = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal problems
        problems += 0 if ok else 1
        print(f"  [{'ok' if ok else 'MISSING'}] {label}" + (f"  {detail}" if detail else ""))

    print("python packages")
    for mod in ("cv2", "numpy", "torch", "easyocr", "h5py", "tqdm", "sklearn"):
        try:
            m = __import__(mod)
            check(mod, True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            check(mod, False, str(e).splitlines()[0])
    try:
        import torch  # noqa: WPS433
        check("CUDA available", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "OCR will run on CPU (slow)")
    except Exception:  # noqa: BLE001
        pass

    print("repository assets")
    for rel in ("models/OCR_interface/simpleocr/best_crnn_model.pth", "data/roi_img_caches",
                "config/rois.json", "config/machine_identity.json"):
        check(rel, (REPO / rel).exists())

    print("google cloud")
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    check("gcloud on PATH", bool(exe), exe or "run scripts/setup.sh or install the SDK")
    if exe:
        acct = gcloud(["auth", "list", "--filter", "status:ACTIVE", "--format", "value(account)"], capture=True)
        active = (acct.stdout or "").strip()
        check("gcloud account", bool(active), active or "run: gcloud auth login --no-browser")
        probe = gcloud(["storage", "ls", cfg["bucket"]], capture=True)
        check(f"bucket readable {cfg['bucket']}", probe.returncode == 0, (probe.stderr or "").strip().splitlines()[-1] if probe.returncode else "")
    print("all good" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


# --------------------------------------------------------------------------- cli

def main(argv: Optional[Sequence[str]] = None) -> int:
    # keep our progress lines in order with gcloud's own output when piped to a log
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", help="gs://bucket (default: config/cloud.json or $SURGERYOCR_GCS_BUCKET)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check packages, GPU, assets, gcloud auth and bucket access")
    sub.add_parser("list", help="show videos and result directories in the bucket")

    p = sub.add_parser("push", help="upload videos (files or directories) to the bucket")
    p.add_argument("paths", nargs="+", help="video files or directories under data/")
    p.add_argument("--force", action="store_true", help="re-upload even if the object exists")

    p = sub.add_parser("pull", help="download videos; --run analyses each one after download")
    p.add_argument("names", nargs="*", help="video name, stem, key relative to data/, or glob")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true", help="re-download even if the local file matches")
    p.add_argument("--run", action="store_true", help="run run_pipeline.py on each video after download")
    p.add_argument("--mode", default="ram", choices=["detail", "frame", "ram", "read"], help="run_pipeline.py --mode")
    p.add_argument("--upload-results", action="store_true", help="with --run: push results to the bucket after each video")

    p = sub.add_parser("push-results", help="upload data/<video>/*.json* for the given videos")
    p.add_argument("names", nargs="*", help="video files or directories")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite result files already in the bucket")

    p = sub.add_parser("pull-results", help="download result directories into data/")
    p.add_argument("names", nargs="*", help="video stem or glob")
    p.add_argument("--all", action="store_true")

    args = ap.parse_args(argv)
    cfg = load_config(args.bucket)
    handler = {
        "doctor": cmd_doctor, "list": cmd_list, "push": cmd_push, "pull": cmd_pull,
        "push-results": cmd_push_results, "pull-results": cmd_pull_results,
    }[args.cmd]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
