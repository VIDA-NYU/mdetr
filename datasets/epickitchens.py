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

from .frames import TextImageDataset


def load_dataset_splits(path='/vast/irr2020/EPIC-KITCHENS', split=None):
    path = Path(path)
    # load as one dataframe
    train_df = pd.read_csv(path/'epic-kitchens-100-annotations/EPIC_100_train.csv')
    val_df = pd.read_csv(path/'epic-kitchens-100-annotations/EPIC_100_validation.csv')
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    df = pd.concat([train_df, val_df])
    if split:
        df = df[df.split == split]
    assert len(df), "No data."
    
    df = df[['split', 'participant_id', 'video_id']].drop_duplicates()

    def get_video_frames_df(row):
        # find common files between video/text
        p, v = row.participant_id, row.video_id
        video_dir = os.path.join(path, 'frames', row.participant_id, row.video_id)
        text_dir = os.path.join(path, 'labels_narration', row.participant_id, row.video_id)
        frame_ids = (
            set(os.path.splitext(f) for f in os.listdir(video_dir)) & 
            set(os.path.splitext(f) for f in os.listdir(text_dir)))
        
        print(row.split, row.participant_id, row.video_id,  f'{len(frame_ids)} files indexed')
        
        return pd.DataFrame({ 
            **row, 
            'frame_id': frame_ids,
            'image_path': [os.path.join(video_dir, f'{fid}.jpg') for fid in frame_ids],
            'text_path': [os.path.join(text_dir, f'{fid}.txt') for fid in frame_ids],
        })
    df = df.apply(get_video_frames_df)
    # now df has one row per frame

    # as dicts
    image_files_train = {f'{row.video_id}/{row.frame_id}': row.image_path for i, row in df[df.split ==  'train'].iterrows()}
    text_files_train = {f'{row.video_id}/{row.frame_id}': row.text_path for i, row in df[df.split ==  'train'].iterrows()}
    image_files_val = {f'{row.video_id}/{row.frame_id}': row.image_path for i, row in df[df.split ==  'val'].iterrows()}
    text_files_val = {f'{row.video_id}/{row.frame_id}': row.text_path for i, row in df[df.split ==  'val'].iterrows()}
    return image_files_train, text_files_train, image_files_val, text_files_val



class EpicKitchens(TextImageDataset):
    def load_lists(self, train):
        image_files_train, text_files_train, image_files_val, text_files_val = load_dataset_splits()
        if train:
            return image_files_train, text_files_train
        return image_files_val, text_files_val