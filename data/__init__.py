import copy
import os
import torch
from .fedTS_mix import build_fed_dataset
from .lianying_jiangsu_mix import create_datasets
from torch.utils.data import DataLoader, DistributedSampler
from .ixi_mix import build_ixi_dataset

def build_server_dataloader(args, mode='val'):
    mask = os.path.join(args.TRANSFORMS.MASK_DIR, args.TRANSFORMS.MASK_FILE[2])
    data_list = os.path.join(args.DATASET.ROOT[2], 'validation.txt')
    data_loader = []
    dataset = build_fed_dataset(data_root=args.DATASET.ROOT[2], list_file=data_list,
                                                             mask_path=mask, mode=mode,
                                                             sample_rate=args.DATASET.SAMPLE_RATE[2],
                                                             crop_size=(320, 320), pattern=args.DATASET.PATTERN[2],
                                                             totalsize=args.DATASET.NUM_SERVER_VAL,
                                                             sample_delta=4
                                                             )
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE, sampler=sampler,
                              num_workers=args.SOLVER.NUM_WORKERS, pin_memory=False, drop_last=False))

    return data_loader, [len(dataset)]


def build_different_dataloader(args, mode='train'):
    mask_path = [os.path.join(args.TRANSFORMS.MASK_DIR, f) for f in args.TRANSFORMS.MASK_FILE]
    crop_size = (args.MODEL.INPUTSIZE, args.MODEL.INPUTSIZE)

    jiangsu_dataset = create_datasets(args, data_root=args.DATASET.ROOT[0],  mode=mode,
                                      sample_rate=args.DATASET.SAMPLE_RATE[0], client_name='JiangSu',
                                      pattern=args.DATASET.PATTERN[0], crop_size=crop_size,
                                      totalsize=args.DATASET.NUM_TRAIN[0] if mode=='train' else args.DATASET.NUM_VAL[0],
                                      slice_range=(0, 19))

    lianying_dataset = create_datasets(args, data_root=args.DATASET.ROOT[1], mode=mode,
                                       sample_rate=args.DATASET.SAMPLE_RATE[1], client_name='lianying',
                                       pattern=args.DATASET.PATTERN[1], crop_size=crop_size,
                                       totalsize=args.DATASET.NUM_TRAIN[1] if mode == 'train' else args.DATASET.NUM_VAL[1],
                                       slice_range=(0, 18))

    fedlist = [3,5,16,20,6,13,21,4,18,1]
    feds = {}
    fed_root = args.DATASET.ROOT[2]
    for idx, sub_client in enumerate(fedlist):
        subidx= idx+2
        train_list = os.path.join(fed_root, f'{mode}_{sub_client:02d}.txt')
        feds[f'fed_dataset{sub_client:02d}'] = build_fed_dataset(data_root=fed_root, list_file=train_list, mask_path=mask_path[subidx], mode=mode,
                                                        sample_rate=args.DATASET.SAMPLE_RATE[subidx],
                                                        crop_size=crop_size, pattern =args.DATASET.PATTERN[subidx],
                                                        totalsize=args.DATASET.NUM_TRAIN[subidx] if mode == 'train' else args.DATASET.NUM_VAL[subidx],
                                                        )

    ixi_totalsize = {'HH':   {'train': args.DATASET.NUM_TRAIN[12], 'val': args.DATASET.NUM_VAL[12]},
                     'Guys': {'train': args.DATASET.NUM_TRAIN[13], 'val': args.DATASET.NUM_VAL[13]},
                     'IOP':  {'train': args.DATASET.NUM_TRAIN[14], 'val': args.DATASET.NUM_VAL[14]}
                     }
    ixi_root = args.DATASET.ROOT[3]
    HH_list = os.path.join(ixi_root, f'HH_{mode}.txt')
    ixi_HH_dataset = build_ixi_dataset(data_root=ixi_root, list_file=HH_list, mask_path=mask_path[12], mode=mode, pattern=args.DATASET.PATTERN[12],
                                       type='HH', sample_rate=args.DATASET.SAMPLE_RATE[12],
                                       totalsize=ixi_totalsize['HH'][mode], crop_size=crop_size)

    Guys_list = os.path.join(ixi_root, f'Guys_{mode}.txt')
    ixi_Guys_dataset = build_ixi_dataset(data_root=ixi_root, list_file=Guys_list, mask_path=mask_path[13], mode=mode, pattern=args.DATASET.PATTERN[13],
                                         type='Guys', sample_rate=args.DATASET.SAMPLE_RATE[13],
                                         totalsize=ixi_totalsize['Guys'][mode], crop_size=crop_size)

    IOP_list = os.path.join(ixi_root, f'IOP_{mode}.txt')
    ixi_IOP_dataset = build_ixi_dataset(data_root=ixi_root, list_file=IOP_list, mask_path=mask_path[14], mode=mode, pattern=args.DATASET.PATTERN[14],
                                        type='IOP', sample_rate=args.DATASET.SAMPLE_RATE[14],
                                        totalsize=ixi_totalsize['IOP'][mode], crop_size=crop_size)

    datasets = [jiangsu_dataset, lianying_dataset, *feds.values(), ixi_HH_dataset, ixi_Guys_dataset, ixi_IOP_dataset]


    data_loader = []
    dataset_len = []
    for dataset in datasets:
        dataset_len.append(len(dataset))
        if mode == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)

            data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE, sampler=sampler,
                                         num_workers=args.SOLVER.NUM_WORKERS, pin_memory=False, drop_last=False))
        elif mode == 'val':
            sampler = torch.utils.data.SequentialSampler(dataset)
            data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE,  sampler=sampler,
                                         num_workers=args.SOLVER.NUM_WORKERS, pin_memory=False, drop_last=False))
    return data_loader, dataset_len


