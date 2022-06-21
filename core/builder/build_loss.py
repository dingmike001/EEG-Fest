import torch


def loss_function(cfg):
    if cfg.train_parameters.loss_function == 'cross_entropy':
        if cfg.train_parameters.loss_weight == 'None':
            weight = None
        else:
            weight = torch.tensor(cfg.train_parameters.loss_weight, dtype=torch.float, device='cuda')
        loss = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        raise NotImplementedError
    return loss
