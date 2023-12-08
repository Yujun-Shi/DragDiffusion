# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import PIL
from PIL import Image

from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import tqdm

import sys
sys.path.insert(0, '../')
from utils.lora_utils import train_lora


if __name__ == '__main__':
    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    # assume root_dir and lora_dir are valid directory
    root_dir = 'drag_bench_data'
    lora_dir = 'drag_bench_lora'

    # mkdir if necessary
    if not os.path.isdir(lora_dir):
        os.mkdir(lora_dir)
        for cat in all_category:
            os.mkdir(os.path.join(lora_dir,cat))

    for cat in all_category:
        file_dir = os.path.join(root_dir, cat)
        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':
                continue
            sample_path = os.path.join(file_dir, sample_name)

            # read image file
            source_image = Image.open(os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)

            # load meta data
            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']

            # train and save lora
            save_lora_path = os.path.join(lora_dir, cat, sample_name)
            if not os.path.isdir(save_lora_path):
                os.mkdir(save_lora_path)

            # you may also increase the number of lora_step here to train longer
            train_lora(source_image, prompt,
                model_path="runwayml/stable-diffusion-v1-5",
                vae_path="default", save_lora_path=save_lora_path,
                lora_step=80, lora_lr=0.0005, lora_batch_size=4, lora_rank=16, progress=tqdm, save_interval=10)
