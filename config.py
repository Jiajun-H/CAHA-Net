import os
import argparse

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))

# 默认路径（相对到仓库根）
DEFAULT_DATA_DIR = os.path.join(ROOT, 'data')
DEFAULT_TRAIN_DIR = os.path.join(DEFAULT_DATA_DIR, 'train')
DEFAULT_VAL_DIR = os.path.join(DEFAULT_DATA_DIR, 'val')
DEFAULT_MODEL_DIR = os.path.join(ROOT, 'model_pth')
DEFAULT_SAVE_MODEL_DIR = os.path.join(ROOT, 'save_model')
DEFAULT_LOG_DIR = os.path.join(ROOT, 'logs')
DEFAULT_PRETRAIN_CACHE = os.path.expanduser('~/.cache/torch/hub/checkpoints')


def get_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--train_dir', default=None)
    parser.add_argument('--val_dir', default=None)
    parser.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--save_model_dir', default=DEFAULT_SAVE_MODEL_DIR)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    parser.add_argument('--pretrain_cache', default=DEFAULT_PRETRAIN_CACHE)

    # parse known args so scripts can still accept other args
    args, _ = parser.parse_known_args()

    if args.train_dir is None:
        args.train_dir = os.path.join(args.data_dir, 'train')
    if args.val_dir is None:
        args.val_dir = os.path.join(args.data_dir, 'val')

    # ensure directories exist (do not create data dirs automatically)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.save_model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args
