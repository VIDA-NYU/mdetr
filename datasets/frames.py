# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint, choice
import pandas as pd
import PIL
import argparse
import clip
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule



class TextImageDataset(Dataset):
    def __init__(self,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=False,
                 train = True,
                 ):
        """Create a text image dataset from a directory with congruent text and image names.
        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()
        self.shuffle = shuffle
        self.image_files, self.text_files = self.load_lists(train)
        self.keys = list(self.image_files)
        
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.Lambda(self.fix_img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.custom_tokenizer = custom_tokenizer
    def __len__(self):
        return len(self.keys)
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, i):
        return self.__getitem__((i + 1) % self.__len__())

    def skip_sample(self, i):
        return self.random_sample() if self.shuffle else self.sequential_sample(i)

    def load_lists(self, train):
        image_files_train, text_files_train, image_files_val, text_files_val = load_dataset_splits()
        if train:
            return image_files_train, text_files_train
        return image_files_val, text_files_val
        
    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]
        try:
            descriptions = text_file.read_text().split('\n')
        except UnicodeDecodeError:
            return self.skip_sample(ind)
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        tokenized_text = description if self.custom_tokenizer else clip.tokenize(description, truncate=True)[0]
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        # Success
        return image_tensor, tokenized_text



class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 num_workers=0,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=None
                 ):
        """Create a text image datamodule from directories with congruent text and image names.
        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.custom_tokenizer = custom_tokenizer
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        #parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--batch_size', type=int, help='size of the batch')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
        parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
        return parser
    
    def setup(self, stage=None):
        self.dataset = TextImageDataset(
            image_size=self.image_size, 
            resize_ratio=self.resize_ratio, 
            shuffle=self.shuffle, 
            custom_tokenizer=not self.custom_tokenizer is None)
        self.dataset_val = TextImageDataset(
            image_size=self.image_size, 
            resize_ratio=self.resize_ratio, 
            shuffle=self.shuffle, 
            custom_tokenizer=not self.custom_tokenizer is None, 
            train=False)
    

    def _dataloader(self, *a, **kw):
        return DataLoader(*a, **{
            **dict(batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True , collate_fn=self.dl_collate_fn),
            **kw
        })
    def val_dataloader(self):
        return self._dataloader(self.dataset_val)
    def train_dataloader(self):
        return self._dataloader(self.dataset)
    
    def dl_collate_fn(self, batch):
        im, txt = torch.stack([row[0] for row in batch]), [row[1] for row in batch]
        if self.custom_tokenizer is not None:
            return im, self.custom_tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
        return im, torch.stack(txt)