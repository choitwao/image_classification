from cli import Cli
from classification import Classification
import os

if __name__ == '__main__':

    if not os.path.isdir('output/figure'):
        os.makedirs('output/figure')
    args = Cli.create_parser().parse_args()
    t = Classification(args.DATASET)
    if args.subparser_name == 'train':
        t.train(args.algorithm)
    else:
        t.predict(args.algorithm)
