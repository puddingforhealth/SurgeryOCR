"""SurgeryOCR-specific adapter; the zettabyte_transfer package stays reusable."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from zettabyte_transfer.cli import main as transfer, remote_path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description='在 Zettabyte IDE 分析影片，成功後可發布至 outbound/')
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--job-id', required=True, help='唯一任務名稱，例如 case001-20261019')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--mode', choices=['ram', 'frame', 'detail'], default='ram')
    p.add_argument('--publish', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--azcopy', default='azcopy')
    args = p.parse_args(argv)
    try:
        job = remote_path(args.job_id)
        if '/' in job:
            raise ValueError('job-id 不可含 /')
        video = args.video.resolve()
        if video.is_file():
            videos = [video]
        elif video.is_dir():
            videos = sorted(x for x in video.iterdir() if x.is_file() and x.suffix.lower() == '.mp4')
        else:
            videos = []
        if not videos or len({v.stem for v in videos}) != len(videos):
            raise ValueError('找不到影片，或影片名稱（不含副檔名）重複。')
        output = args.output_root.resolve() / job
        # Exclusive job directory avoids publishing stale or mixed results.
        output.mkdir(parents=True, exist_ok=False)
        metadata = {'job_id': job, 'videos': [v.name for v in videos],
                    'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
        def save():
            (output / 'job.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        save()
        try:
            with (output / 'analysis.log').open('w', encoding='utf-8') as log:
                for source in videos:
                    command = [sys.executable, '-u', str(ROOT / 'run_pipeline.py'),
                               '--video', str(source), '--output-dir', str(output), '--mode', args.mode]
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{source.name} 分析失敗（exit {result.returncode}），請查看 analysis.log。')
                    analysis = output if output.name == source.stem else output / source.stem
                    stage = json.loads((analysis / 'stage_analysis.json').read_text(encoding='utf-8'))
                    if not stage.get('total_frames') or not list(analysis.glob('*_ocr_testing.jsonl')):
                        raise RuntimeError(f'{source.name} 缺少有效的分析結果。')
            metadata['status'] = 'complete'
            metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
            save()
        except BaseException:
            metadata['status'] = 'failed'
            save()
            raise
        print(f'分析完成：{output}')
        if args.publish:
            options = ['--azcopy', args.azcopy]
            if args.config:
                options += ['--config', str(args.config)]
            return transfer(options + ['publish', str(output), 'results/' + job])
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'錯誤：{exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
