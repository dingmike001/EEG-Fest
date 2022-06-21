import torch


def optimizer_fn(function_type, model, lr=2e-5, momentum=0.9):
    if function_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif function_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    return optimizer
