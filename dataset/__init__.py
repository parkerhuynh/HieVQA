import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# from dataset.vqa_dataset import vqa_dataset
from dataset.vqa_dataset_simpsons import VQADataset as vqa_dataset_simpsons
from dataset.vqa_dataset_vqav2 import VQADataset as vqa_dataset_vqav2
from dataset.randaugment import RandomAugment

dataset_func = {
    "simpsonsvqa": vqa_dataset_simpsons,
    "vqav2": vqa_dataset_vqav2
    
}


def create_dataset(args, istrain=True):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    vqa_dataset = dataset_func[args.dataset]
    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(args.model_config['img_size'], scale=(0.5, 1.0),
                                     interpolation=Image.BICUBIC),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.model_config['img_size'], args.model_config['img_size']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])


    if args.checkpoint == "":
        train_dataset = vqa_dataset(args, train_transform_wohflip, split='train')
        val_dataset = vqa_dataset(args, test_transform, split='val')
        
        return train_dataset, val_dataset
    else:
        vqa_test_dataset =  vqa_dataset(args, test_transform, split='test')
        return [vqa_test_dataset]


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, args, istrain=True):
    if args.checkpoint != "":
        test_dataset = datasets[0]
        # test_sampler = samplers[0]
        test_sampler = None
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            num_workers=4,
            pin_memory=True,
            sampler=test_sampler,
            shuffle=False,
            drop_last=False,
        )
        return test_loader
    else:
        train_dataset, val_dataset = datasets
        train_samples, val_samples = samplers
        train_shuffle = (train_samples is None)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size_train,
            num_workers=4,
            pin_memory=True,
            sampler=train_samples,
            shuffle=train_shuffle,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_test,
            num_workers=4,
            pin_memory=True,
            sampler=val_samples,
            shuffle=False,
            drop_last=False,
        )