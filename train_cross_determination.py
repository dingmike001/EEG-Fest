import argparse, os, yaml, shutil
from core.builder.build_model import CrossAttention_flatten
from core.builder.build_dataset import few_shot_dataset_binary_builder
from core.builder.build_loss import loss_function
from core.builder.build_optimizer import optimizer_fn
from core.engine.cross_attention_train_engine import cross_trainer
from easydict import EasyDict
from utils.logger import setup_logger, logger
import logging
import torch
import warnings
import json

warnings.filterwarnings('ignore')
torch.manual_seed(3407)


def main(args):
    with open(args.configs, 'r') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    # config.config_combine(cfg, args)

    if cfg.train_parameters.use_gpu:
        device = 'cuda'
        assert cfg.train_parameters.num_gpus == 1, 'Current Not Support Multi-GPU Parallel Computing'  # future work to add
    else:
        device = 'cpu'

    run_path = cfg.train_parameters.save_dir

    if args.resume == False:
        print('training a new model...')
        if os.path.isdir(run_path) is False:
            os.makedirs(run_path)

        if args.running_name == 'default':
            from datetime import datetime
            now = datetime.now()
            parent_path = os.path.join(run_path, args.running_name + '-' + now.strftime("%b-%d-%Y-%H:%M:%S"))
        else:
            parent_path = os.path.join(run_path, args.running_name)
        if os.path.exists(parent_path):
            shutil.rmtree(parent_path)
        os.mkdir(parent_path)
        log_path = os.path.join(parent_path, 'logging_files')
        model_path = os.path.join(parent_path, 'models')
        os.mkdir(log_path)
        os.mkdir(model_path)
    else:
        print('resuming from last training...')
        if args.running_name == 'default':
            from datetime import datetime
            now = datetime.now()
            parent_path = os.path.join(run_path, args.running_name + '-' + now.strftime("%b-%d-%Y-%H:%M:%S"))
        else:
            from datetime import datetime
            now = datetime.now()
            parent_path = os.path.join(run_path, args.running_name)
            log_path = os.path.join(parent_path, 'logging_files' + '-' + now.strftime("%b-%d-%Y-%H:%M:%S"))
            model_path = os.path.join(parent_path, 'models')
            os.mkdir(log_path)

    print('save the logger to: ' + log_path)
    print('save the model to: ' + model_path)

    trainlogfilename = os.path.join(log_path, cfg.dataset_parameters.dataset_name + '_train.log')
    evallogfilename = os.path.join(log_path, cfg.dataset_parameters.dataset_name + '_eval.log')
    setup_logger('train', trainlogfilename)
    setup_logger('eval', evallogfilename)
    logger(['Name:', args.running_name, 'Dataset: ', cfg.dataset_parameters.dataset_name], 'info', 'train')
    logger(['Name:', args.running_name, 'Dataset: ', cfg.dataset_parameters.dataset_name], 'warning', 'eval')
    logger('start training...', 'info', 'train')


    few_shot_test_dataloader, few_shot_val_dataloader = few_shot_dataset_binary_builder(cfg, device)
    model = CrossAttention_flatten(cfg, device)
    model = model.to(device)
    loss = loss_function(cfg)
    optimizer = optimizer_fn(cfg.train_parameters.optimizer, model, lr=cfg.train_parameters.learn_rate, momentum=0.9)
    cross_trainer(few_shot_test_dataloader, few_shot_val_dataloader, model, loss, optimizer,
                  model_path, cfg.train_parameters.epoch_num, log_path)

    print('Training Done!')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for EEG Feature Extraction Algorithms')
    parser.add_argument('--configs',
                        default='./configs/feature_extraction_algorithms/cross_attention_vig.yaml',
                        help='configuration files')
    parser.add_argument('--resume', default=False, help='resume training from last breakpoint')
    parser.add_argument('--running-name', default='cross_attention_train_1',
                        help='running folder to save log files and models')
    args = parser.parse_args()
    main(args)
