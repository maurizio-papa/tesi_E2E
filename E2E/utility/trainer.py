from detector.TriDet.libs.core import load_config
from detector.TriDet.libs.datasets import make_dataset, make_data_loader
from detector.TriDet.libs.modeling import make_meta_arch
from detector.TriDet.libs.utils import  (save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)
                         
import time, copy
import os
import torch
import datetime
import pickle

def train_one_epoch(model, optimizer,  scheduler, data_loader, cfg, optimizer=None, scheduler=None):   #TODO add epochs

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    max_iteration = len(data_loader)
    cfg = load_config(cfg)

    for video_data in enumerate(data_loader):

        video_data = video_data.to(DEVICE)

        # stage 1: sequentially forward the backbone
        video_feat = model(video_data['feats'], stage=1)

        # stage 2: forward and backward the detector
        video_feat.requires_grad = True
        video_feat.retain_grad()

        losses =  model(video_data, stage= 2)

        # backward the detector
        optimizer.zero_grad()
        losses['final_loss'].backward()
        optimizer.step()
        scheduler.step()

        # stage 3: sequentially backward the backbone with sampled data
        feat_grad = copy.deepcopy(video_feat.grad.detach())  # [B,C,T]

        # sample snippets and sequentially backward
        optimizer.zero_grad()
        model(video_data, feat_grad=feat_grad, stage=3)
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

    save_checkpoint(model, epoch, scheduler, optimizer)


def save_checkpoint(model, epoch, cfg, scheduler, optimizer):

    state = {
        "epoch": epoch,
        "state_dict": model.module.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint_dir = "./exps/%s/checkpoint/" % (exp_name)

    if not os.path.exists(checkpoint_dir):
        os.system("mkdir -p %s" % (checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_%d.pth.tar" % epoch)
    torch.save(state, checkpoint_path)