from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry, profile_events_dataframe
from hybrid_schedule.data.profiling import render_profile_markdown



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = load_registry(args.registry)
    df = load_all_events(registry, timezone_default=cfg['calendar']['timezone_default'])
    profile = profile_events_dataframe(df)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dataset_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'dataset_profile.md').write_text(render_profile_markdown(profile), encoding='utf-8')
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
