import logging

import os


# def setup_logger(cfg, log_path, mode, level=logging.INFO):
#     if mode == 'train':
#         name = '_train.log'
#     else:
#         name = '_val.log'
#     logging.basicConfig(filename=os.path.join(log_path, cfg.dataset_parameters.dataset_name + name),
#                         filemode='w',
#                         format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S',
#                         level='NOTSET')
#     logger = logging.getLogger(__name__)
#     logger.setLevel(level)
#     return logger


def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


def logger(msg, level, logfile):
    if logfile == 'train': log = logging.getLogger('train')
    if logfile == 'eval': log = logging.getLogger('eval')
    if level == 'info': log.info(msg)
    if level == 'warning': log.warning(msg)
    if level == 'error': log.error(msg)
