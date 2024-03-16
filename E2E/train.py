import torch 

from utility.loader import EpicKitchenLoader
from build_model import TAL_model
from utility.trainer import train_one_epoch
from detector.TriDet.libs.core import load_config
from detector.TriDet.libs.utils import  (save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, ModelEma)


cfg_optimizer =  {
                        # solver
                        "type": "AdamW",  # SGD or AdamW
                        # solver params
                        "momentum": 0.9,
                        "weight_decay": 0.0,
                        "learning_rate": 1e-3,
                        # excluding the warmup epochs
                        "epochs": 30,
                        # lr scheduler: cosine / multistep
                        "warmup": True,
                        "warmup_epochs": 5,
                        "schedule_type": "cosine",
                        # Minimum learning rate
                        "eta_min": 1e-8,
                        # in #epochs excluding warmup
                        "schedule_steps": [],
                        "schedule_gamma": 0.1,
                 }



def train(cfg, epochs = 8):

    cfg = load_config(cfg)

    # build dataset and dataloader
    train_dataset = EpicKitchenLoader(feature_folder = 'Z:\tensor', 
                                      json_folder = 'Z:\annotations\epic_kitchens_100_noun.json',
                                      split = 'training',
                                      num_frames = '16',
                                      feat_stride = 8)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 2,
        num_workers= 1,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # build model 
    model = TAL_model(chunk_size = 4, sampling_ratio = 0.3)

    # optimizer
    optimizer = make_optimizer(model, cfg_optimizer)
     
    # schedule
    num_iters_per_epoch = len(train_loader) 
    scheduler = make_scheduler(optimizer, cfg_optimizer, num_iters_per_epoch)    

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler)


if __name__ == "__main__":
    train('config.yaml')