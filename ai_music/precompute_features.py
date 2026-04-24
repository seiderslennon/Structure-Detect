from pathlib import Path
import argparse

import yaml
from tqdm import tqdm

from ai_music.data.dataset import AudioDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "ai_music/configs/SpecTTTra.yaml"),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "feature_cache"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--deterministic-start",
        action="store_true",
        help="Disable random clip start while caching.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
    data_config = dict(configs["data"])

    data_config["feature_cache_dir"] = args.cache_dir
    data_config["use_feature_cache"] = True
    data_config["write_feature_cache"] = True
    if args.deterministic_start:
        data_config["random_sample"] = False

    for split in args.splits:
        dataset = AudioDataset(data_config, split)
        for i in tqdm(range(len(dataset)), desc=f"Precompute {split}"):
            _ = dataset[i]

    print(f"Done. Cached features in: {args.cache_dir}")


if __name__ == "__main__":
    main()
