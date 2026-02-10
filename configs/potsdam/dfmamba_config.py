from torch.utils.data import DataLoader
from dfmamba.losses import *
from dfmamba.datasets.potsdam_dataset import *
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("fuser_attention_module", "dfmamba/models/dfmamba.py")
fuser_attention_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fuser_attention_module)
TCMDFMConvNeXtMamba = fuser_attention_module.TCMDFMConvNeXtMamba
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 2
val_batch_size = 1
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES
image_size = 1024
crop_size = int(512*float(image_size/1024))

weights_name = "dfmamba-"+str(image_size)+"-e"+str(max_epoch)
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "dfmamba-"+str(image_size)+"-e"+str(max_epoch)
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained DFMamba weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

# Model specific parameters
EMBED_DIM = 64
DECODE_CHANNELS = 64
FUSION_CHANNELS = [64, 128, 256, 512]
PRETRAINED_CNN = False
PRETRAINED_TRANS = True
CNN_BACKBONE_PATH = None
TRANS_BACKBONE_PATH = 'pretrain_weights/rest_lite.pth'
CONVNEXT_WEIGHT_PATH = 'pretrain_weights/convnext_tiny.pth'  # 使用timm自动下载或设为None
DEBUG_SHAPES = False
# Contrastive learning parameters (enabled for Loss+Attention version)
CONTRAST_ENABLE = True
CONTRAST_PROJ_DIM = 128
CONTRAST_TAU = 0.07
CONTRAST_STAGE_WEIGHTS = (0.1, 0.2, 0.3, 0.4)
CONTRAST_PIXEL = False
CONTRAST_PIXEL_NUM = 128

#  define the network
net = TCMDFMConvNeXtMamba(
    pretrained_cnn=PRETRAINED_CNN,
    pretrained_trans=PRETRAINED_TRANS,
    cnn_backbone_path=CNN_BACKBONE_PATH,
    trans_backbone_path=TRANS_BACKBONE_PATH,
    convnext_weight_path=CONVNEXT_WEIGHT_PATH,
    embed_dim=EMBED_DIM,
    decode_channels=DECODE_CHANNELS,
    num_classes=num_classes,
    fusion_channels=FUSION_CHANNELS,
    debug_shapes=DEBUG_SHAPES,
    contrast_enable=CONTRAST_ENABLE,
    contrast_proj_dim=CONTRAST_PROJ_DIM,
    contrast_tau=CONTRAST_TAU,
    contrast_stage_weights=CONTRAST_STAGE_WEIGHTS,
    contrast_pixel=CONTRAST_PIXEL,
    contrast_pixel_num=CONTRAST_PIXEL_NUM
)

# define the loss
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=crop_size, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask
  

train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               img_dir='images_'+str(image_size), mask_dir='masks_'+str(image_size),
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = PotsdamDataset(data_root='data/potsdam/test', 
                             img_dir='images_'+str(image_size), mask_dir='masks_'+str(image_size),
                             transform=val_aug)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              img_dir='images_'+str(image_size), mask_dir='masks_'+str(image_size),
                              transform=val_aug)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True,
                          persistent_workers=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
