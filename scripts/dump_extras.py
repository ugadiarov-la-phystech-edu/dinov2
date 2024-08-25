from argparse import ArgumentParser
import dinov2.data.datasets

parser = ArgumentParser()
parser.add_argument('--dataset_class', type=str, required=True)
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--extra_path', type=str, required=True)


def main(args):
    dataset_class = getattr(dinov2.data.datasets, args.dataset_class)
    for split in dataset_class.Split:
        dataset = dataset_class(split=split, root=args.root_path, extra=args.extra_path)
        dataset.dump_extra()


if __name__ == "__main__":
    main(parser.parse_args())