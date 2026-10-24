"""Resolve a viewer's machine from the analysis it displays, without editing mappings."""
import json
from pathlib import Path
import re

from utils.get_configs import get_machine_key, get_video_machine_id


def analysis_machine(analysis_dir, video_name, roi_file):
    """Prefer result metadata, then legacy pipeline evidence, then local configuration."""
    root = Path(analysis_dir)
    source = root / '_source.json'
    if source.is_file():
        machine = json.loads(source.read_text(encoding='utf-8')).get('machine_id')
        if machine is not None:
            if str(machine) not in ('1', '2', '3'):
                raise ValueError('Unsupported analysis machine_id')
            return int(machine), '分析結果'
    log = root / 'pipeline.log'
    if log.is_file():
        machines = set()
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            if any(marker in line for marker in ('偵測機型:', '檢測到已知設定:', '使用預設 Type')):
                match = re.search(r'\bType\s+([123])\b', line)
                if match:
                    machines.add(int(match.group(1)))
        if len(machines) > 1:
            raise ValueError('Conflicting machine IDs in analysis log')
        if machines:
            return machines.pop(), '分析日誌'
    machine = get_video_machine_id(video_name, Path(roi_file))
    if machine is not None:
        return int(machine), '本機設定'
    return 1, '預設（無機型紀錄）'


def machine_config(config_root, filename, machine_id):
    root = Path(config_root)
    candidate = root / get_machine_key(machine_id) / filename
    return candidate if candidate.is_file() else root / filename
