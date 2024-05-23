from argparse import ArgumentParser
from dinov2.data.datasets import MiniImageNet 

parser = ArgumentParser()



def main():
    for split in MiniImageNet.Split:
        dataset = MiniImageNet(split=split, root="/code/data/imagenet-mini", extra="/code/data/imagenet-mini/extra")
        dataset.dump_extra()
        
if __name__ == "__main__":
    main()