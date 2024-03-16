from TriDet.libs.core import load_config
from TriDet.libs.modeling import meta_archs
from TriDet.libs.datasets import make_dataset, make_data_loader
from TriDet.libs.modeling import TriDet
from TriDet.libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

def load_detector(config):

    model = TriDet(
            backbone_type = 'SGP',  # a string defines which backbone we use
            fpn_type  = 'identity',  # a string defines which fpn we use
            backbone_arch = (2, 2, 5),  # a tuple defines # layers in embed / stem / branch
            scale_factor = 2,  # scale factor between branch layers
            input_dim = 2304,  # input feat dim
            max_seq_len = 2304,  # max sequence length (used for training)
            max_buffer_len_factor = 2304,  # max buffer size (defined a factor of max_seq_len)
            n_sgp_win_size = 1,  # window size w for sgp
            embd_kernel_size = 3,  # kernel size of the embedding network
            embd_di = 512,  # output feat channel of the embedding network
            embd_with_ln = True,  # attach layernorm to embedding network
            fpn_dim = 512,  # feature dim on FPN,
            sgp_mlp_dim = 1024,  # the numnber of dim in SGP
            fpn_with_ln = True,  # if to apply layer norm at the end of fpn
            head_dim = 512,  # feature dim for head
            regression_range = [[ 0, 4 ], [ 2, 8 ], [ 4, 16 ], [ 8, 32 ], [ 16, 64 ], [ 32, 10000 ]],  # regression range on each level of FPN
            head_num_layers = 3,  # number of layers in the head (including the classifier)
            head_kernel_size = 3,  # kernel size for reg/cls heads
            boudary_kernel_size = 3,  # kernel size for boundary heads
            head_with_ln = True,  # attache layernorm to reg/cls heads
            use_abs_pe = False,  # if to use abs position encoding
            num_bins = 15,  # the bin number in Trident-head (exclude 0)
            iou_weight_power = 0.25,  # the power of iou weight in loss
            downsample_type = 'max',  # how to downsample feature in FPN
            input_noise = 0,  # add gaussian noise with the variance, play a similar role to position embedding
            k = 4,  # the K in SGP
            init_conv_vars = 0,  # initialization of gaussian variance for the weight in SGP
            use_trident_head = True,  # if use the Trident-head
            num_classes = 97,  # number of action classes
            train_cfg = "train_cfg": {
                                                "center_sample": "radius",
                                                "center_sample_radius": 1.5,
                                                "loss_weight": 1.0,  # on reg_loss, use -1 to enable auto balancing
                                                "cls_prior_prob": 0.01,
                                                "init_loss_norm": 2000,
                                                # gradient cliping, not needed for pre-LN transformer
                                                "clip_grad_l2norm": -1,
                                                # cls head without data (a fix to epic-kitchens / thumos)
                                                "head_empty_cls": [],
                                                # dropout ratios for tranformers
                                                "dropout": 0.0,
                                                # ratio for drop path
                                                "droppath": 0.1,
                                                # if to use label smoothing (>0.0)
                                                "label_smoothing": 0.0,
                                      },  
        test_cfg =     "test_cfg": {
                                                "pre_nms_thresh": 0.001,
                                                "pre_nms_topk": 5000,
                                                "iou_threshold": 0.1,
                                                "min_score": 0.01,
                                                "max_seg_num": 1000,
                                                "nms_method": 'soft',  # soft | hard | none
                                                "nms_sigma": 0.5,
                                                "duration_thresh": 0.05,
                                                "multiclass_nms": True,
                                                "ext_score_file": None,
                                                "voting_thresh": 0.75,})
        

    return model 