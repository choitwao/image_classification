from cli import Cli
from train import Train

if __name__ == '__main__':

    args = Cli.create_parser().parse_args()
    if args.subparser_name == 'train':
        train = Train(args.DATASET)
