#!/usr/bin/env python3
"""Run legacy/enhanced pipelines separately and score their visible value timelines."""

import argparse
import base64
from bisect import bisect_right
import html
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ocr_runtime import file_sha256, load_bundle  # noqa: E402
from utils.ocr_training import numerical_metrics, load_dataset, contained_path  # noqa: E402


def value_timeline(path):
    """Expand cache matches; a deactivation wins at an equal frame index."""
    events = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            deactivate = record.get("type") == "deactivate"
            text = "" if deactivate else record.get("ocr_text", "")
            frames = set(record.get("matched_frames", []))
            start = record.get("source_frame", record.get("frame"))
            if start is not None:
                frames.add(start)
            for frame in frames:
                priority = 1 if deactivate else 0
                if frame not in events or priority >= events[frame][0]:
                    events[frame] = (priority, text)
    frames = sorted(events)
    return frames, [events[frame][1] for frame in frames]


def value_at(timeline, frame):
    frames, texts = timeline
    index = bisect_right(frames, frame) - 1
    return texts[index] if index >= 0 else ""


def metrics(rows):
    result = {}
    for profile in ["legacy", "enhanced"]:
        pairs = [(r["label"], r[profile]) for r in rows]
        result[profile] = {
            **numerical_metrics(pairs),
            "blank_false_positives": sum(a == "" and b != "" for a, b in pairs),
            "missing_visible_values": sum(a != "" and b == "" for a, b in pairs),
        }
    result["improved"] = [
        r["id"]
        for r in rows
        if r["legacy"] != r["label"] and r["enhanced"] == r["label"]
    ]
    result["regressed"] = [
        r["id"]
        for r in rows
        if r["legacy"] == r["label"] and r["enhanced"] != r["label"]
    ]
    return result


def load_plan(path):
    videos = json.loads(Path(path).read_text(encoding="utf-8"))
    ids, stems = set(), set()
    for video in videos:
        identity = video["video_id"]
        stem = Path(video["source_path"]).stem.casefold()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identity) is None:
            raise ValueError("Video IDs must be opaque filename-safe identifiers")
        if identity in ids or stem in stems:
            raise ValueError(
                "Duplicate video ID or output filename stem in comparison plan"
            )
        if video["machine_id"] not in {1, 2, 3}:
            raise ValueError("Comparison machine_id must be 1, 2 or 3")
        ids.add(identity)
        stems.add(stem)
    if not videos:
        raise ValueError("Empty comparison plan")
    return videos


def stage_rule_allows(stage_analysis, activation, region, frame):
    allowed = activation.get(region, [])
    if not allowed:
        return True
    pattern = next(
        (
            s["pattern"]
            for s in stage_analysis.get("regions", {}).get("STAGE", [])
            if s["start_frame"] <= frame <= s["end_frame"]
        ),
        -1,
    )
    return pattern in allowed


def run_plan(args):
    videos = load_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    for video in videos:
        source = Path(video["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        for profile in ["legacy", "enhanced"]:
            directory = args.output / profile / source.stem
            if directory.exists():
                statistics = directory / "run_stats.json"
                if not args.reuse_complete or not statistics.exists():
                    raise ValueError(
                        f"Refuse to overwrite incomplete/existing results: {directory}"
                    )
                data = json.loads(statistics.read_text(encoding="utf-8"))
                if data.get("ocr", {}).get("profile") != profile:
                    raise ValueError("Existing result profile does not match")
                if (
                    int(data.get("ocr", {}).get("machine_id", -1))
                    != video["machine_id"]
                ):
                    raise ValueError("Existing result uses a different machine layout")
                if profile == "enhanced":
                    _, bundle = load_bundle(args.bundle)
                    provenance = data["ocr"]
                    if provenance.get("bundle_version") != bundle[
                        "version"
                    ] or provenance.get("weights_sha256") not in {
                        a["sha256"] for a in bundle["artifacts"].values()
                    }:
                        raise ValueError(
                            "Existing result uses a different OCR model bundle"
                        )
                    if (
                        video["machine_id"] in {1, 2}
                        and provenance.get("sensitivity") != args.sensitivity
                    ):
                        raise ValueError(
                            "Existing result uses a different OCR trigger sensitivity"
                        )
                print("Reusing completed run:", video["video_id"], profile, flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "run_pipeline.py"),
                "--video",
                str(source),
                "--output-dir",
                str(args.output / profile),
                "--mode",
                "ram",
                "--ocr-profile",
                profile,
                "--ocr-device",
                args.device,
                "--ocr-sensitivity",
                args.sensitivity,
                "--machine-id",
                str(video["machine_id"]),
            ]
            if args.bundle:
                command.extend(["--ocr-bundle", str(args.bundle)])
            print("Running:", video["video_id"], profile, flush=True)
            with (args.output / f"{video['video_id']}-{profile}.log").open(
                "w", encoding="utf-8"
            ) as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("Completed:", video["video_id"], profile, flush=True)


def report(args):
    videos = load_plan(args.plan)
    manifest12 = json.loads(
        (args.machine12_data / "dataset.json").read_text(encoding="utf-8")
    )
    rows12 = [
        r
        for r in load_dataset(args.machine12_data, accept_assistant=True)
        if r["split"] == "test"
    ]
    rows3 = [
        json.loads(line)
        for line in (args.machine3_data / "labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions, runs = [], []
    for video in videos:
        source = Path(video["source_path"])
        paths = {p: args.runs / p / source.stem for p in ["legacy", "enhanced"]}
        stats = {
            p: json.loads((d / "run_stats.json").read_text(encoding="utf-8"))
            for p, d in paths.items()
        }
        stage_equal = file_sha256(
            paths["legacy"] / "stage_analysis.json"
        ) == file_sha256(paths["enhanced"] / "stage_analysis.json")
        clock_equal = file_sha256(paths["legacy"] / "frame_clock.json") == file_sha256(
            paths["enhanced"] / "frame_clock.json"
        )
        if (
            not stage_equal
            or not clock_equal
            or stats["legacy"]["frames"] != stats["enhanced"]["frames"]
        ):
            raise ValueError(
                f"Stage, clock or frame-count regression: {video['video_id']}"
            )
        rows = [
            r
            for r in (rows3 if video["machine_id"] == 3 else rows12)
            if r["video_id"] == video["video_id"]
        ]
        timelines = {}
        activation = json.loads(
            (ROOT / "config/ocr_activation_stages.json").read_text(encoding="utf-8")
        )
        stage_data = json.loads(
            (paths["enhanced"] / "stage_analysis.json").read_text(encoding="utf-8")
        )
        review_path = paths["enhanced"] / "ocr_review.jsonl"
        reviews = (
            [
                json.loads(line)
                for line in review_path.read_text(encoding="utf-8").splitlines()
            ]
            if review_path.exists()
            else []
        )
        for row in rows:
            for profile, directory in paths.items():
                key = (profile, row["region"])
                if key not in timelines:
                    timelines[key] = value_timeline(
                        directory / (row["region"] + "_ocr_testing.jsonl")
                    )
            root = (
                args.machine3_data if video["machine_id"] == 3 else args.machine12_data
            )
            relative = (
                row["image"] if video["machine_id"] == 3 else row["images"]["rgb"]
            )
            image_path = contained_path(root, relative)
            predictions.append(
                {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "machine_id": video["machine_id"],
                    "region": row["region"],
                    "frame": row["frame"],
                    "label": row["label"],
                    "image": str(image_path.resolve()),
                    "stage_rule_allows": stage_rule_allows(
                        stage_data, activation, row["region"], row["frame"]
                    ),
                    "enhanced_review": next(
                        (
                            r
                            for r in reviews
                            if r["region"] == row["region"]
                            and r["frame"]
                            <= row["frame"]
                            <= r.get("end_frame", r["frame"])
                        ),
                        None,
                    ),
                    **{
                        p: value_at(timelines[(p, row["region"])], row["frame"])
                        for p in paths
                    },
                }
            )
        runs.append(
            {
                "video_id": video["video_id"],
                "machine_id": video["machine_id"],
                "stage_equal": stage_equal,
                "frame_clock_equal": clock_equal,
                "statistics": stats,
            }
        )
    summary = {
        "scope": "Full sequential decoded-video pipeline runs; annotated-frame value lookup includes trigger, activation, cache and blank behavior. Model-development videos and assistant labels are not untouched acceptance data.",
        "machine12_ledger_sha256": file_sha256(
            args.machine12_data / manifest12.get("manifest", "records.jsonl")
        ),
        "machine3_ledger_sha256": file_sha256(args.machine3_data / "labels.jsonl"),
        "metrics": metrics(predictions),
        "by_machine": {
            str(m): metrics([r for r in predictions if r["machine_id"] == m])
            for m in sorted({r["machine_id"] for r in predictions})
        },
        "runs": runs,
        "predictions": predictions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cells = []
    for row in predictions:
        image = base64.b64encode(Path(row["image"]).read_bytes()).decode()
        different = row["legacy"] != row["enhanced"] or row["enhanced"] != row["label"]
        text = [
            f"{row['video_id']} / {row['region']} / F{row['frame']}",
            row["label"] or "空白",
            row["legacy"] or "空白",
            row["enhanced"] or "空白",
        ]
        if (
            not row["enhanced"]
            and row["enhanced_review"]
            and row["enhanced_review"].get("raw_text")
        ):
            text[-1] = "待覆核：" + row["enhanced_review"]["raw_text"]
        if not row["stage_rule_allows"]:
            if not row["legacy"]:
                text[-2] = "未啟用（階段規則）"
            if not row["enhanced"]:
                text[-1] = "未啟用（階段規則）"
        cells.append(
            f'<tr data-focus="{str(different).lower()}"><td><img src="data:image/png;base64,{image}" alt="crop"></td>'
            + "".join("<td>" + html.escape(s) + "</td>" for s in text)
            + "</tr>"
        )
    tables = []
    for machine, data in summary["by_machine"].items():
        a, b = data["legacy"], data["enhanced"]
        tables.append(
            f"<tr><td>機型 {machine}</td><td>{a['exact']}/{a['samples']}</td><td>{b['exact']}/{b['samples']}</td><td>{len(data['improved'])}</td><td>{len(data['regressed'])}</td></tr>"
        )
    page = (
        """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>新舊分析流程比較</title><style>body{font:16px system-ui;background:#101b28;color:#e2edf6;margin:28px}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:12px;border-bottom:1px solid #345;text-align:left}th{background:#183047}img{height:80px;max-width:280px;object-fit:contain}label{display:block;margin:20px 0}.note{max-width:1000px;color:#bacbdc}</style><h1>新舊分析流程比較</h1><p class="note">每列是一個已判讀畫面。只看「正確值、舊流程、新流程」即可；空白表示畫面沒有數值。以下是整部影片跑完後，查詢同一幀當時實際輸出的值，包含啟動條件、變化偵測、快取及清空舊值的影響。</p><p class="note">標註由助理判讀；機型3的兩部影片已用於模型開發。這是整合回歸比較，不是新的獨立臨床或跨影片驗收。</p><table><tr><th>機型</th><th>舊流程正確</th><th>新流程正確</th><th>改善案例</th><th>退步案例</th></tr>"""
        + "".join(tables)
        + """</table><p>所有比較影片的階段分析、時間軸與解碼幀數均須一致，否則報告工具會停止。</p><label><input type="checkbox" id="focus" checked>只看有變化或新流程仍錯誤的案例</label><table><thead><tr><th>畫面</th><th>位置</th><th>正確值</th><th>舊流程</th><th>新流程</th></tr></thead><tbody id="samples">"""
        + "".join(cells)
        + """</tbody></table><script>const focusBox=document.getElementById('focus');function apply(){document.querySelectorAll('#samples tr').forEach(r=>r.hidden=focusBox.checked&&r.dataset.focus!=='true')}focusBox.addEventListener('change',apply);apply();</script></html>"""
    )
    timing_rows = []
    for run in runs:
        old, new = run["statistics"]["legacy"], run["statistics"]["enhanced"]
        ratio = (
            new["total_seconds"] / old["total_seconds"] if old["total_seconds"] else 0
        )
        timing_rows.append(
            f"<tr><td>{html.escape(run['video_id'])}</td><td>{new['frames']:,}</td><td>{old['total_seconds']:.1f} 秒</td><td>{new['total_seconds']:.1f} 秒</td><td>{ratio:.2f} 倍</td></tr>"
        )
    timing = (
        '<h2>完整影片耗時</h2><p class="note">包含解碼與分析；桌面背景負載未控制。新機型3路線逐幀檢查RGB，工作量較大，準確度改善不等於速度變快。</p><table><tr><th>影片</th><th>解碼幀數</th><th>舊流程耗時</th><th>新流程耗時</th><th>新／舊時間</th></tr>'
        + "".join(timing_rows)
        + "</table>"
    )
    page = page.replace("<label><input", timing + "<label><input", 1)
    page = page.replace(
        "空白表示畫面沒有數值。",
        "正確值欄的空白表示畫面沒有數字；流程欄的空白表示當時沒有輸出數值。待覆核會保留原始讀值，不會自行補負號或小數點。",
    )
    suppressed = sum(
        not r["stage_rule_allows"] and not r["legacy"] and not r["enhanced"]
        for r in predictions
    )
    if suppressed:
        note = f'<p class="note">另有 {suppressed} 個標註畫面位於原有階段規則的停用時段，兩流程都沒有輸出。報告保留可見數值和這些案例的分母，因此上表衡量的是整條流程與畫面的數值相符程度，不只辨識器準確率；未將停用案例排除以提高分數。</p>'
        page = page.replace("<p>所有比較影片", note + "<p>所有比較影片", 1)
    page = page.replace("舊流程正確", "舊流程相符").replace("新流程正確", "新流程相符")
    (args.output / "comparison.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Private JSON list: source_path, video_id, machine_id",
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bundle", type=Path)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--sensitivity", choices=["standard", "exact"], default="standard")
    run.add_argument("--reuse-complete", action="store_true")
    show = commands.add_parser("report")
    for flag in ["plan", "runs", "output", "machine12-data", "machine3-data"]:
        show.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    run_plan(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    main()
