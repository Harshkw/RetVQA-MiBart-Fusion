from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from Get_hdf5 import load_image_embed
from Get_hdf5 import load_question

class CustomDataset:
    def __init__(self, json_file, image_folder, device, num_images = 2048, split = "train"):
        self.json_file = json_file
        self.image_folder = image_folder
        self.device = device
        self.num_samples = num_images
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        # Filter data based on the split
        if self.split == "total":
            return {k: v for k, v in data.items()}
        else:
            return {k: v for k, v in data.items() if v['split'] == self.split}

    def __len__(self):
        item = list(self.data.values())
        return len(item)

    def upload_image(self, image_list, images_type = "positive"):
        image_tensors = []
        if images_type == "negative":
            target_count = 23
        else:
            target_count = 2
        img_counter = 0

        for image_id in image_list:
            try:
                image = load_image_embed("/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/RetVQA_Data_HDF5/images_embed_HDF5", str(image_id))
                image_tensors.append(image.to(self.device))
                img_counter += 1
                if img_counter == target_count:
                    break

            except FileNotFoundError:
                print(f"Image {image_id}.jpg not found in {self.image_folder}")
                continue

        while img_counter < target_count:
            for image_id in image_list:
                try:
                    image = load_image_embed("/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/RetVQA_Data_HDF5/images_embed_HDF5", str(image_id))
                    image_tensors.append(image.to(self.device))
                    img_counter += 1
                    if img_counter == target_count:
                        break
                except FileNotFoundError:
                    continue 
        return image_tensors        
    
    def __getitem__(self, idx):
        qid = list(self.data.keys())[idx]
        sample = list(self.data.values())[idx]
        positive_img_ids = sample.get('pos_imgs', [])
        negative_img_ids = sample.get('neg_imgs', [])
        total_ids = positive_img_ids + negative_img_ids

        # Loading the images from Dataset
        positive_image_encoding = self.upload_image(positive_img_ids)        
        negative_image_encoding = self.upload_image(negative_img_ids, images_type = "negative")
        
        # Tokeninzing the question text
        question_encoding = load_question("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/questions_HDF5",qid).to(self.device)
            
        positive_image_encoding = torch.stack(positive_image_encoding).squeeze(1)  
        negative_image_encoding = torch.stack(negative_image_encoding).squeeze(1)
        h=2
        while(len(total_ids)<25):
            total_ids.append(total_ids[h])
            h+=1        
        while len(total_ids)>25:
            total_ids.pop()
        # total_ids.append(qid)
        return question_encoding, positive_image_encoding, negative_image_encoding, torch.tensor(total_ids),qid