from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("--splits", '-s', nargs="+", default=["train", "val"])




def main(args):
    results = {}
    for split in args.splits:
        dataset_path: Path = args.path / split
        assert dataset_path.exists(), f"Split '{split}' does not exists"
        results[split] = len(list(dataset_path.rglob('*.JPEG')))    
    
    print("Results of scanning")
    for split, count in results.items():
        print(f'{split}: {count}')


if __name__ == "__main__":
    main(parser.parse_args())