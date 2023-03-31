from yacs.config import CfgNode as CN
import os


_C = CN()
_C.SEED = 42

_C.dist_url = 'env://'
_C.world_size = 1

# _C.MU = 10
_C.lam = 1
_C.beta = 100

_C.DATASET = CN()
_C.DATA_ROOT = os.path.join(os.path.expanduser('~'), 'data/AAA-Download-Datasets')
_C.DATASET.ROOT = [
    os.path.join(_C.DATA_ROOT, 'Fed/Data/JiangSu'),
    os.path.join(_C.DATA_ROOT, 'Fed/Data/lianying'),
    os.path.join(_C.DATA_ROOT, 'Fed/Data/FeTS2022'),
    os.path.join(_C.DATA_ROOT, 'Fed/Data/IXI-NIFTI'),
]

_C.DATASET.CLIENTS = ['JiangSu', 'lianying', 'FedTS01', 'FedTS02', 'FedTS03', 'FedTS04', 'FedTS05', 'FedTS06',
                      'FedTS07', 'FedTS08', 'FedTS09', 'BraTS10', 'ixiHH', 'ixiGuys', 'ixiIOP']
_C.DATASET.PATTERN = ['T2', ] * 15
_C.DATASET.SAMPLE_RATE = [1, ] * 15
_C.DATASET.CHALLENGE = 'singlecoil'


_C.DATASET.NUM_TRAIN = [432, 360, 120, 304, 304, 240, 160, 160, 280, 224, 264, 184, 360, 328, 296]
_C.DATASET.NUM_VAL =   [184, 154, 80,  130, 130, 100, 64,  64,  120, 96,  112, 78,  152, 120, 128]
_C.DATASET.NUM_SERVER_VAL = 1712


_C.TRANSFORMS = CN()
_C.TRANSFORMS.MASK_FILE = ["1D-Cartesian_3X_320.mat"] * 15

_C.TRANSFORMS.MASK_DIR = _C.DATA_ROOT + '/Fed/Data/masks'
# _C.TRANSFORMS.MASK_SPEC = ['2D-Radial-4X', '2D-Random-6X']
_C.TRANSFORMS.MASK_SPEC = ["1D-Cartesian_3X", ] * 15


_C.FL = CN()
_C.FL.CLIENTS_NUM = 15
_C.FL.MODEL_NAME = 'swin_vpt_nullspace'
_C.FL.SHOW_SIZE = True

# model config
_C.MODEL = CN()
_C.MODEL.TRANSFER_TYPE = "prompt"  # "prompt"  # one of linear, end2end, prompt, adapter, side, partial-1, tinytl-bias
_C.MODEL.WEIGHT_PATH = None
_C.MODEL.SUBTYPE = "swin_320"
_C.MODEL.INPUTSIZE = 320
_C.MODEL.FINALSIZE = 320
_C.MODEL.HEAD_NUM_CONV = 4
_C.MODEL.INPUT_DIM = 1
_C.MODEL.OUTPUT_DIM = 1
_C.MODEL.PROMPT = CN()
_C.MODEL.PROMPT.NUM_TOKENS = 20
_C.MODEL.PROMPT.LOCATION = "prepend"
_C.MODEL.PROMPT.INITIATION = "random"
_C.MODEL.PROMPT.PROJECT = -1
_C.MODEL.PROMPT.DEEP = True  # "whether do deep prompt or not, only for prepend location"
_C.MODEL.PROMPT.DROPOUT = 0.0

# the solver config

_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda:1'
_C.SOLVER.DEVICE_IDS = [0, 1, 2, 3]
_C.SOLVER.LR = [0.1, ] * 15
_C.SOLVER.WEIGHT_DECAY = 0
_C.SOLVER.LR_DROP = 20
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.RATIO = 0.8

# the others config
_C.RESUME = ''  # model resume path
_C.PRETRAINED_FASTMRI_CKPT = 'path/to/pretrained/fastmri_ckpt.pth'
_C.OUTPUTDIR = './saved'


#the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS =       50
_C.TRAIN.LOCAL_EPOCHS = 10

_C.meta_client_num = 15
_C.DISTRIBUTION_TYPE = 'in-distribution'  #  'out-of-distribution'  # 'in-distribution'


