import torch
import numpy as np
import os
from utils.postprocess import get_accuracy
from utils.logger import logger


def evaluator(val_data_loader, epoch, model, loss, curr_accuracy, logger, model_path):
    print('evaluation...')
    logger.info(['evaluating...'])
    model.eval()
    pred_cls_sum = []
    gt_cls_sum = []
    pred_loss_sum = []
    with torch.no_grad():
        for idx, content in enumerate(val_data_loader):
            data = torch.tensor(content['samples']).to('cuda').float()
            target = torch.tensor(content['labels']).to('cuda').long()
            gt_cls_sum.append(target.item())
            predict = model(data)
            pred_loss_sum.append(loss(predict.squeeze(), target.squeeze()).item())
            _, pred_cls = torch.max(predict, dim=1)
            pred_cls_sum.append(pred_cls.item())
        avg_loss = np.mean(pred_loss_sum)
        accuracy = get_accuracy(pred_cls_sum, gt_cls_sum, 'weighted')
        logger.info(
            ['epoch:', epoch, 'average_loss: ', avg_loss, 'average_accuracy: ', accuracy['total_accuracy'] * 100])
        if accuracy['total_accuracy'] > curr_accuracy:
            curr_accuracy = accuracy['total_accuracy']
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best.pth'))

        print('****average accuracy is: %3f, average loss is: %5f ****' % (accuracy['total_accuracy'], avg_loss))
        print(accuracy['class_accuracies'])


def few_shot_evaluator(val_data_loader, epoch, model, loss, curr_accuracy, logger, model_path):
    print('evaluation...')
    logger.info(['evaluating...'])
    model.eval()
    pred_cls_sum = []
    gt_cls_sum = []
    pred_loss_sum = []
    with torch.no_grad():
        for idx, content in enumerate(val_data_loader):
            s_samples = content['s_sample']
            q_samples = content['q_sample']
            s_targets = content['s_target']
            q_targets = content['q_target']
            gt_cls_sum.append(q_targets.item())
            predict = model(s_samples, q_samples, s_targets, q_targets)
            predict = predict.unsqueeze(0)
            q_targets = torch.tensor(q_targets, dtype=torch.long, device='cuda')
            pred_loss_sum.append(loss(predict, q_targets).item())
            _, pred_cls = torch.max(predict, dim=1)
            pred_cls_sum.append(pred_cls.item())
        avg_loss = np.mean(pred_loss_sum)
        accuracy = get_accuracy(pred_cls_sum, gt_cls_sum, 'weighted')
        logger.info(
            ['epoch:', epoch, 'average_loss: ', avg_loss, 'average_accuracy: ', accuracy['total_accuracy'] * 100,
             'accuracy_matrix: ', accuracy['class_accuracies']])
        if accuracy['total_accuracy'] > curr_accuracy:
            curr_accuracy = accuracy['total_accuracy']
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best.pth'))

        print('****average accuracy is: %3f, average loss is: %5f ****' % (accuracy['total_accuracy'], avg_loss))
        print(accuracy['class_accuracies'])
        return curr_accuracy, avg_loss


def few_shot_binary_evaluator(val_data_loader, epoch, model, loss, curr_accuracy, model_path):
    print('evaluation...')
    logger('evaluating...', 'warning', 'eval')
    model.eval()
    pred_cls_sum = []
    gt_cls_sum = []
    pred_loss_sum = []


    with torch.no_grad():
        for idx, content in enumerate(val_data_loader):
            s_samples = content['s_sample']
            q_samples = content['q_sample']
            s_targets = content['s_target']
            q_targets = content['q_target']
            cross_targets = content['cross_target']
            predict = model(s_samples, q_samples, s_targets, q_targets)
            q_targets = torch.tensor(q_targets, dtype=torch.long, device='cuda')
            if q_targets.shape[0] > 1:
                q_targets = q_targets.squeeze()
            gt_cls_sum.append(q_targets.squeeze().detach().to('cpu').numpy().tolist())
            pred_loss_sum.append(loss(predict, q_targets).item())
            _, pred_cls = torch.max(predict, dim=1)
            pred_cls_sum.append(pred_cls.detach().to('cpu').numpy().tolist())
        avg_loss = np.mean(pred_loss_sum)
        accuracy = get_accuracy(pred_cls_sum, gt_cls_sum, 'weighted')
        logger(['epoch:', epoch, 'average_loss: ', avg_loss, 'average_accuracy: ', accuracy['total_accuracy'] * 100,
                'accuracy_matrix: ', accuracy['class_accuracies']], 'warning', 'eval')
        if accuracy['total_accuracy'] > curr_accuracy:
            curr_accuracy = accuracy['total_accuracy']
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best.pth'))

        print('****average accuracy is: %3f, average loss is: %5f ****' % (accuracy['total_accuracy'], avg_loss))
        print(accuracy['class_accuracies'])
        return curr_accuracy, avg_loss
