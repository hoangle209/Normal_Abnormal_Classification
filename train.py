import torch
import torch.nn as nn

from models.trainer import Trainer
from dataset.dataset import GenericDataset
from models.models import Classify
from models.utils import load_model, save_model
from opts import opts

import os


def get_model(opt):
    if opt.model == 'YoloAtt':
        nc = 2 if opt.loss == 'CE' else 1
        model = Classify(nc=nc)
    
    return model


def get_optimizer(opt, model):
    if opt.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == 'SGD':
        optimizer = torch.optim.SGD(
        model.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    else:
        assert 0, opt.optim

    return optimizer


def main(opt):
    torch.manual_seed(opt.seed)
    # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = GenericDataset(opt.path, opt)
    train_loader = torch.utils.data.DataLoader(
        Dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    if opt.val > 0:
        Dataset_val = GenericDataset(opt.val_path, opt)
        val_loader = torch.utils.data.DataLoader(
            Dataset_val, batch_size=1, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, drop_last=True
        )

    # module_list = nn.ModuleList([])

    model = get_model(opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    if opt.load_model != '' and opt.resume:
        model, optimizer, start_epoch = load_model(model, opt.load_model, opt, optimizer)
    elif opt.load_model != '':
        model = load_model(model, opt.load_model, opt)
    
    trainer = Trainer(model, opt, optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.set_device(device)

    print('Start training ...')
    print(opt)
    for epoch in range(start_epoch+1, opt.num_epochs+1):
        ret, _ = trainer.train(epoch, train_loader)
        
        if opt.val > 0 and epoch % opt.val:
            with torch.no_grad():
                trainer.val(epoch, val_loader)
        
        save_model(os.path.join(opt.save_path, 'model_last.pth'), epoch, model, optimizer) # save model last


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
        



