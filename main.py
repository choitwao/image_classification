from cli import Cli
from classification import Classification

if __name__ == '__main__':

    args = Cli.create_parser().parse_args()
    t = Classification(args.DATASET)
    if args.subparser_name == 'train':
        t.train(args.algorithm)
    else:
        t.predict(args.algorithm)
