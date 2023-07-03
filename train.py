import torch
import torch.nn as nn

from libs.trainer import Trainer
from libs.backbone.utils import create_model, load_model, save_model
from libs.dataset.dataset_factory import get_dataset
from opts import opts

import os


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
    dataset = get_dataset(opt.dataset)

    Dataset_train = dataset(opt.path, opt, split='train')
    train_loader = torch.utils.data.DataLoader(
        Dataset_train, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    if opt.val > 0:
        Dataset_val = dataset(opt.val_path, opt, split='val')
        val_loader = torch.utils.data.DataLoader(
            Dataset_val, batch_size=1, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, drop_last=True
        )

    # module_list = nn.ModuleList([])
    nums_in = Dataset_train.num_rgb + Dataset_train.num_flow + Dataset_train.num_heat
    model = create_model(opt, nums_in=nums_in) # TODO
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
        
        if opt.val > 0 and epoch % opt.val == 0:
            with torch.no_grad():
                trainer.val(epoch, val_loader)
        
        save_model(os.path.join(opt.save_path, f'model_{opt.model_name}_last.pth'), epoch, model, optimizer) # save model last

        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

