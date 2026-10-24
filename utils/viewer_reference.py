"""Read-only comparison against the template recorded for the displayed frame."""
from pathlib import Path
import cv2
import numpy as np
from utils.cv_processing import binarize_pil, calculate_rmse
from utils.get_configs import get_machine_key, load_diff_rules
from utils.viewer_context import machine_config


def region_segments(analysis, region):
    value = (analysis or {}).get('regions', {}).get(region, [])
    return value.get('patterns', []) if isinstance(value, dict) else value


def recorded_segment(analysis, region, frame):
    return next((s for s in region_segments(analysis, region)
                 if s['start_frame'] <= frame <= s['end_frame']), None)


class ReferenceChecker:
    def __init__(self, config_root, cache_root):
        self.config_root, self.cache_root = Path(config_root), Path(cache_root)
        self.rules, self.references = {}, {}

    def inspect(self, image, machine_id, region, pattern):
        route = get_machine_key(machine_id)
        key = (route, region, int(pattern))
        path = self.cache_root/route/region/f'{int(pattern)}.npy'
        if route not in self.rules:
            self.rules[route] = load_diff_rules(machine_config(self.config_root, 'diff_rule.json', machine_id))
        rule = self.rules[route].get(region, {})
        threshold = rule.get('diff_threshold', 30.0)
        if key not in self.references:
            if not path.is_file():
                return {'state': 'unavailable', 'threshold': threshold}
            self.references[key] = np.load(path, allow_pickle=False)
        reference = self.references[key]
        rgb = np.asarray(image.convert('RGB'))
        if reference.ndim == 3:
            candidate = rgb
            if rule.get('analysis_mode') == 'sub_roi':
                x1,y1,x2,y2 = rule.get('sub_roi_coords', [20,13,26,19])
                if x1 < 0 or y1 < 0 or x2 > rgb.shape[1] or y2 > rgb.shape[0]:
                    return {'state': 'unavailable', 'threshold': threshold}
                candidate, reference = rgb[y1:y2,x1:x2], reference[y1:y2,x1:x2]
        elif rule.get('binarize') == 'rule':
            candidate = binarize_pil(image.convert('RGB'))
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            _, candidate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if candidate.shape != reference.shape or candidate.size == 0:
            return {'state': 'unavailable', 'threshold': threshold}
        score = float(calculate_rmse(candidate.astype(np.float32), reference.astype(np.float32)))
        return {'state': 'match' if score < threshold else 'different', 'rmse': score, 'threshold': threshold}
