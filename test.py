import argparse
import logging

from data_preprocessing import CachedEmbedLoader
from train import RobertaTrainer, WSDNetXTrainer, RDenseTrainer, WSDDenseTrainer
from utils.config import RobertaTransformerConfig, WSDNetXConfig, RDenseConfig, WSDDenseConfig
from utils.util import telegram_on_failure

ALL_TESTS = ['se2', 'se3', 'se07', 'se13', 'se15']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train with different models and options")
    parser.add_argument("-m", "--model", type=str, help="model name",
                        required=True, choices=('roberta', 'wsdnet', 'wsdnetx', 'rdense', 'wsddense'))
    parser.add_argument("-c", "--config", type=str, help="config JSON file path", required=True)
    parser.add_argument("-d", "--debug", action='store_true', help="Print debug information")
    parser.add_argument("-g", "--multi-gpu", action='store_true', help="Use all available GPUs.")
    parser.add_argument("-l", "--log", type=str, help="log file name")
    parser.add_argument("-z", "--cache", type=str, help="Embeddings cache", default='res/cache')
    parser.add_argument("-a", "--all", type=str, help="Test on all dataset. Provide root folder.",
                        default="res/wsd-test/")
    args = parser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.log:
        logging.basicConfig(filename=args.log, level=log_level, format='%(asctime)s: %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=log_level, format='%(asctime)s: %(levelname)s: %(message)s')
    logging.info(f'Initializing... model = {args.model}')
    if args.config.endswith('_half.json'):
        BATCH_MUL = CachedEmbedLoader.HALF
    c, t = None, None
    if args.model == 'roberta':
        c = RobertaTransformerConfig.from_json_file(args.config)
    elif args.model == 'wsdnetx':
        c = WSDNetXConfig.from_json_file(args.config)
    elif args.model == 'rdense':
        c = RDenseConfig.from_json_file(args.config)
    elif args.model == 'wsddense':
        c = WSDDenseConfig.from_json_file(args.config)
    cd = c.__dict__
    cd['is_training'] = False
    cd['multi_gpu'] = args.multi_gpu
    cd['cache_path'] = args.cache
    logging.info(f"Results on concatenation of test dataset:")
    if args.model == 'roberta':
        t = RobertaTrainer(**cd)
    elif args.model == 'wsdnetx':
        t = WSDNetXTrainer(**cd)
    elif args.model == 'rdense':
        t = RDenseTrainer(**cd)
    elif args.model == 'wsddense':
        t = WSDDenseTrainer(**cd)
    telegram_on_failure(t.test)

    for dataset in ALL_TESTS:
        logging.info(f"Results on dataset {dataset}:")
        cd['test_data'] = f"{args.all}/{dataset}/{dataset}.xml"
        cd['test_tags'] = f"{args.all}/{dataset}/{dataset}.txt"
        if args.model == 'roberta':
            t = RobertaTrainer(**cd)
        elif args.model == 'wsdnetx':
            t = WSDNetXTrainer(**cd)
        elif args.model == 'rdense':
            t = RDenseTrainer(**cd)
        elif args.model == 'wsddense':
            t = WSDDenseTrainer(**cd)
        telegram_on_failure(t.test)



