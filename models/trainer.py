import torch
import torch.nn as nn

from .losses import BCELoss, CELoss
from ..utils.utils import AverageMeter

from progress.bar import Bar

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model
        self.crit_label = CELoss(label_smoothing=1./opt.batch_size)
        self.opt = opt


    def _sigmoid_ouput(self, out):
        out = torch.clamp(out.sigmoid_(), min=1e-4, max=1-1e-4)

        return out

    def _softmax_output(self, out):
        out = nn.functional.softmax(out, axis=-1)

        return out
    

    def forward(self, batch):
        out = self.model(batch['image'])

        if self.opt.loss == 'CE':
            out = self._softmax_output(out)
        elif self.opt.loss == 'BCE':
            out = self._sigmoid_ouput(out)
        else:
            raise NotImplementedError

        loss = self.crit_label(out, batch['label'].detach())
        loss_stat = {
            'label': loss
        }

        return out, loss, loss_stat


class Trainer():
    def __init__(self, model, opt, optimizer=None):
        self.optimizer = optimizer
        self.opt = opt
        self.model_with_loss = ModelWithLoss(model, opt)
    
    def set_device(self, device):
        # if len(gpus) > 1:
        #     self.model_with_loss = DataParallel(
        #         self.model_with_loss, device_ids=gpus, 
        #         chunk_sizes=chunk_sizes).to(device)
        # else:
        
        self.model_with_loss = self.model_with_loss.to(device)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        elif phase == 'val':
            model_with_loss.eval()
            torch.cuda.empty_cache()

        avg_loss_stats = {'label': AverageMeter()}

        max_iter = len(dataset)
        bar = Bar(f'Abnormal Classification', max=max_iter)

        if phase == 'val':
            TP = 0

        for idx, batch in enumerate(dataset):
            out, loss, loss_stat = self.model_with_loss(batch)
            
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stat[l].mean().item(), batch['image'].size(0)
                )
                Bar.suffix = Bar.suffix + f'|{l} {avg_loss_stats[l].avg:.4f} '
            Bar.suffix = f'[{phase}][{epoch}][{idx}/{max_iter}]|Tot: {bar.elapsed_td:} |ETA: {bar.eta_td:} '

            if phase == 'val':
                pred = torch.argmax(out).cuda().int()
                gt = torch.argmax(batch['label']).cuda().int()
                
                if pred == gt:
                    TP += 1
            bar.next()
        bar.finish()

        if phase == 'val':
            print(f'\nValid Precision epoch: [{epoch}]: {TP/max_iter}')

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        
        return ret, None


    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)