import torch
from tqdm import tqdm, trange
from core.engine.eval_engine import evaluator
import itertools
from sklearn.metrics import accuracy_score

from utils.OptimScheduler import GradualWarmupScheduler



def trainer(train_data_loader, val_data_loader, model, loss, optimizer, logger, model_path,epoch_num):
    curr_accuracy = 0.0

    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    optim_scheduler_warm = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10,
                                                  after_scheduler=optim_scheduler)
    for epoch in range(epoch_num):
        logger.info(['training...'])
        model.train()
        iter_loss = 0
        train_out = []
        gt_out = []
        with trange(len(train_data_loader), unit="iteration", desc='epoch ' + str(epoch)) as pbar:
            for idx, content in enumerate(train_data_loader):
                data = torch.tensor(content['samples']).to('cuda').float()
                target = torch.tensor(content['labels']).to('cuda').long()

                predict = model(data)
                train_out.append(torch.max(predict, dim=1)[1].detach().to('cpu').numpy().tolist())
                gt_out.append(target.squeeze().detach().to('cpu').numpy().tolist())
                criterion = loss(predict, target.squeeze())
                optimizer.zero_grad()
                criterion.backward()
                optimizer.step()
                iter_loss += criterion.item()
                pbar.set_postfix(loss=criterion.item())
                pbar.n += 1
            avg_loss = iter_loss/(idx+1)
            optim_scheduler_warm.step(epoch)

            predict_out = list(itertools.chain.from_iterable(train_out))
            target_out = list(itertools.chain.from_iterable(gt_out))
            acc_avg = accuracy_score(predict_out, target_out)
            logger.info(['epoch:', epoch, 'average_loss: ', avg_loss, 'average_accuracy: ', acc_avg*100])
            evaluator(val_data_loader, epoch, model, loss, curr_accuracy, logger, model_path)

