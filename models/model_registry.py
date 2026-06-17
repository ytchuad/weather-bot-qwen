# models/model_registry.py
"""模型版本註冊與比較工具"""
import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

REGISTRY_PATH = Path('models/model_registry.json')
ARCHIVE_DIR = Path('models/archive')

def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {'models': []}

def save_registry(registry):
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)

def register_new_version(model_name, metrics, artifacts, training_config):
    """註冊新模型版本，若優於現行則設為 active"""
    registry = load_registry()
    # 找出該模型最新的 active 版本
    active = None
    for m in registry['models']:
        if m['name'] == model_name and m.get('status') == 'active':
            active = m
            break

    version_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_entry = {
        'name': model_name,
        'version': version_id,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'training_config': training_config,
        'status': 'candidate'
    }

    # 若無 active 或新模型更優，則設為 active，舊版改為 archived
    promote = False
    if active is None:
        promote = True
    else:
        # 比較：以 Pinball Loss 總和為主要指標，越低越好
        old_loss = sum(active['metrics'].get('pinball_loss', {}).values())
        new_loss = sum(metrics.get('pinball_loss', {}).values())
        if new_loss < old_loss:
            promote = True

    if promote:
        new_entry['status'] = 'active'
        if active:
            active['status'] = 'archived'
    else:
        new_entry['status'] = 'archived'

    # 複製模型檔案至 archive
    version_dir = ARCHIVE_DIR / f"{model_name}_{version_id}"
    version_dir.mkdir(parents=True, exist_ok=True)
    for src in artifacts:
        shutil.copy2(src, version_dir / Path(src).name)
    new_entry['artifact_path'] = str(version_dir)

    registry['models'].append(new_entry)
    save_registry(registry)
    return promote, new_entry