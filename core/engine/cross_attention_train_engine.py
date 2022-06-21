import torch
from tqdm import trange
import itertools
from sklearn.metrics import accuracy_score
import os
from utils.OptimScheduler import GradualWarmupScheduler
from utils.postprocess import get_accuracy
from core.engine.eval_engine import few_shot_binary_evaluator
import json
from torch.nn.utils import clip_grad_norm_
from utils.logger import logger
from matplotlib import pyplot as plt


def cross_trainer(few_shot_test_loader, few_shot_val_loader, model, loss, optimizer,
                  model_path, epoch_num, log_path):
    curr_accuracy = 0.0
    loss_all = []
    accuracy_all = []
    val_loss_all = []
    val_accuracy_all = []
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    optim_scheduler_warm = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10,
                                                  after_scheduler=optim_scheduler)

    for epoch in range(epoch_num):
        # train_logger.info(['training...'])
        logger('training...', 'info', 'train')
        model.train()
        iter_loss = 0
        train_out = []
        gt_out = []

        with trange(len(few_shot_test_loader), unit="iteration", desc='epoch ' + str(epoch)) as pbar:
            for idx, content in enumerate(few_shot_test_loader):
                s_samples = content['s_sample']
                q_samples = content['q_sample']
                s_targets = content['s_target']
                q_targets = content['q_target']
                cross_targets = content['cross_target']
                predict = model(s_samples, q_samples, s_targets, q_targets)
                [train_out.append(ele) for ele in
                 torch.max(predict, dim=-1)[1].squeeze().detach().to('cpu').numpy().tolist()]

                q_targets = torch.tensor(q_targets, dtype=torch.long, device='cuda')
                # cross_targets = torch.transpose(cross_targets, 1, 0)

                if q_targets.shape[0] > 1:
                    q_targets = q_targets.squeeze()
                if predict.dim() < 2:
                    predict = predict.unsqueeze(0)

                criterion = loss(predict, q_targets)
                optimizer.zero_grad()
                criterion.backward()
                # clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                [gt_out.append(ele) for ele in q_targets.squeeze().detach().to('cpu').numpy().tolist()]
                iter_loss += criterion.item()
                pbar.set_postfix(loss=criterion.item())
                pbar.n += 1
            avg_loss = iter_loss / (idx + 1)
            loss_all.append(avg_loss)
            optim_scheduler_warm.step(epoch)

            # predict_out = list(itertools.chain.from_iterable(train_out))
            # target_out = list(itertools.chain.from_iterable(gt_out))
            predict_out = train_out
            target_out = gt_out
            acc_avg = accuracy_score(predict_out, target_out)
            accuracy_all.append(acc_avg)
            accuracy = get_accuracy(predict_out, target_out, 'weighted')

            logger(
                ['epoch:', epoch, 'average_loss: ', avg_loss, 'average_accuracy: ', acc_avg * 100, 'accuracy_matrix: ',
                 accuracy['class_accuracies']], 'info', 'train')

            curr_accuracy, val_loss = few_shot_binary_evaluator(few_shot_val_loader, epoch, model, loss, curr_accuracy,
                                                                model_path)

            val_loss_all.append(val_loss)
            val_accuracy_all.append(curr_accuracy)
    json_train_data = {'average_loss': loss_all, 'average_accuracy': accuracy_all}
    json_data_string = json.dumps(json_train_data)
    train_log_path = os.path.join(log_path, 'train_result.json')
    with open(train_log_path, 'w') as outfile:
        outfile.write(json_data_string)
    train_jpg_path = os.path.join(log_path, 'train_loss.jpg')
    plt.figure(1)
    plt.plot(loss_all)
    plt.title('train_loss')
    plt.savefig(train_jpg_path)
    train_jpg_path = os.path.join(log_path, 'train_accuracy.jpg')
    plt.figure(2)
    plt.plot(accuracy_all)
    plt.title('train_accuracy')
    plt.savefig(train_jpg_path)
    json_val_data = {'average_loss': val_loss_all, 'average_accuracy': val_accuracy_all}
    json_data_string = json.dumps(json_val_data)
    val_log_path = os.path.join(log_path, 'val_result.json')
    with open(val_log_path, 'w') as outfile:
        outfile.write(json_data_string)
    train_jpg_path = os.path.join(log_path, 'val_loss.jpg')
    plt.figure(3)
    plt.plot(val_loss_all)
    plt.title('evaluate_loss')
    plt.savefig(train_jpg_path)
    train_jpg_path = os.path.join(log_path, 'val_accuracy.jpg')
    plt.figure(4)
    plt.plot(val_accuracy_all)
    plt.title('evaluate_accuracy')
    plt.savefig(train_jpg_path)
