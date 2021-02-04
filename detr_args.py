# num_classes = 67
num_classes = 67
lr = 1e-05
# lr = 0.0001
lr_backbone = 1e-05
# lr_backbone = 1e-05
batch_size = 32
weight_decay = 0.0001
epochs = 300
lr_drop = 50
clip_max_norm = 0.1
frozen_weights = None
backbone = 'resnet50'
dilation = False
position_embedding = 'sine'
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1
nheads = 8
num_queries = 68
pre_norm = False
masks = False
aux_loss = False
set_cost_class = 1
set_cost_coords = 1
set_cost_giou = 0
mask_loss_coef = 0
dice_loss_coef = 0
label_loss_coef = 0
bbox_loss_coef = 10
giou_loss_coef = 0
eos_coef = 0.1
dataset_file = 'WS02'
coco_path = '/home/itamar/thesis/DATASET/WS02'
coco_panoptic_path = None
remove_difficult = False
output_dir = '/home/itamar/thesis/outputs/detr'
device = 'cuda'
seed = 42
resume = False
start_epoch = 0
eval = False
num_workers = 2
world_size = 1
dist_url = 'env://'
