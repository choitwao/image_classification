from argparse import ArgumentParser

class Cli:

    @staticmethod
    def create_parser():
        # create the main parser for CLI
        command_parser = ArgumentParser(prog='Image Classification')
        # create branches for GET and POST
        method_parsers = command_parser.add_subparsers(help='[command] help',
                                                       dest='subparser_name')
        method_parsers.required = True
        # create a general template for methods
        template_parser = ArgumentParser(add_help=False,
                                         conflict_handler='resolve')
        template_parser.add_argument('-a',
                                     dest='algorithm',
                                     action='store',
                                     metavar='{DT,MB}',
                                     help='Specify the training method.')
        template_parser.add_argument('DATASET',
                                     action='store',
                                     help='The name of the data set (i.e. ds1 for \'data/ds1\'')
        # train
        train_parser = method_parsers.add_parser('train',
                                                 parents=[template_parser],
                                                 help='Train with data set.')
        # predict
        predict_parser = method_parsers.add_parser('predict',
                                                   parents=[template_parser],
                                                   help='Predict with model.')
        predict_parser.add_argument('-o',
                                    dest='',
                                    action='store_const',
                                    const=True,
                                    default=False,
                                    help='Parameter for saving the prediction output as CSV file (default as False).')
        return command_parser




