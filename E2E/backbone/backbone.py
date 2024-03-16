from collections import OrderedDict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from backbone.AVION.avion.data.clip_dataset import get_downstream_dataset
from backbone.AVION.avion.data.tokenizer import tokenize
from backbone.AVION.avion.data.transforms import Permute

import backbone.AVION.avion.models.model_clip as model_clip
from backbone.AVION.avion.models.utils import inflate_positional_embeds
from backbone.AVION.avion.optim.schedulers import cosine_scheduler
import backbone.AVION.avion.utils.distributed as dist_utils


def load_backbone():

    ckpt_path = '.....'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))

    model = getattr(model_clip, old_args.model)(
            freeze_temperature=True,
            use_grad_checkpointing= old_args.use_grad_checkpointing,
            context_length= old_args.context_length,
            vocab_size= old_args.vocab_size,
            patch_dropout= old_args.patch_dropout,
            num_frames= old_args.clip_length,
            drop_path_rate= old_args.drop_path_rate,
            use_fast_conv1= old_args.use_fast_conv1,
            use_flash_attn=old_args.use_flash_attn,
            use_quick_gelu= True,
            project_embed_dim= old_args.project_embed_dim,
            pretrain_zoo=old_args.pretrain_zoo,
            pretrain_path=old_args.pretrain_path,
        )
    model.logit_scale.requires_grad = False
    print('=> inflating PE in models due to different frame numbers')
    state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames= old_args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))

    model = model_clip.VideoClassifier(
            model.visual,
            dropout= old_args.dropout_rate,
            num_classes= old_args.num_classes
        )
    return model

