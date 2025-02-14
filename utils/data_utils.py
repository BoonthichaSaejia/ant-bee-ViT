import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os

logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Data augmentation and normalization for training and testing
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the hymenoptera dataset (bees and ants)
    train_dir = os.path.join(args.dataset_dir, 'train')  # Specify path to your 'train' folder
    val_dir = os.path.join(args.dataset_dir, 'val')  # Specify path to your 'val' folder

    trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    testset = datasets.ImageFolder(root=val_dir, transform=transform_test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Set up samplers for distributed training if necessary
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader
