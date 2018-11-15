from cli import Cli
from classification import Classification

if __name__ == '__main__':

    args = Cli.create_parser().parse_args()
    if args.subparser_name == 'train':
        t = Classification(args.DATASET)
        t.train(args.algorithm)
