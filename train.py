import logging
import os

from torch.utils.tensorboard import SummaryWriter
from train_videosumm import train_videosumm
from config import *
from utils import *

import nltk
nltk.download('punkt')



logger = logging.getLogger()

# For SumMe/TVSum datasets
def main_videosumm(args):
    init_logger(args.model_dir, args.log_file)
    set_random_seed(args.seed)
    dump_yaml(vars(args), '{}/args.yml'.format(args.model_dir))

    logger.info(vars(args))
    os.makedirs(args.model_dir, exist_ok=True)
    print(args.model_dir)

    args.writer = SummaryWriter(os.path.join(args.model_dir, 'tensorboard'))
    
    split_path = '{}/{}/splits.yml'.format(args.data_root, args.dataset)
    split_yaml = load_yaml(split_path)

    f1_results = {}
    f1_results_train = {}
    stats = AverageMeter('fscore')

    for split_idx, split in enumerate(split_yaml):
        logger.info(f'Start training on {split_path}: split {split_idx}')
        max_val_fscore, best_val_epoch, max_train_fscore = train_videosumm(args, split, split_idx) #best_val_epoch이랑 max_train_fscore?
        stats.update(fscore=max_val_fscore)

        f1_results_train[f'split{split_idx}'] = float(max_train_fscore)
        f1_results[f'split{split_idx}'] = float(max_val_fscore)

    logger.info(f'Training done on {split_path}.')
    logger.info(f'F1_results_trian: {f1_results_train}')
    logger.info(f'F1_results_val: {f1_results}')
    logger.info(f'F1-score: {stats.fscore:.4f}\n\n')

if __name__ == '__main__':
    args = get_arguments()
    if args.dataset in ['TVSum', 'SumMe']:
        main_videosumm(args)
    else:
        raise NotImplementedError

