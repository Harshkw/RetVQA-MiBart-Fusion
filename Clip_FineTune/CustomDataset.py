from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from Get import load_image
from Get import load_question
class DatasetPreparation:
    def __init__(self, json_file, image_folder, clip_preprocess, bert_tokenizer, device, num_images = 2048, split = "train"):
        self.json_file = json_file
        self.image_folder = image_folder
        self.device = device
        self.counter = num_images + 1
        self.bert_tokenizer = bert_tokenizer
        self.clip_preprocess = clip_preprocess
        self.num_samples = num_images
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        # Filter data based on the split
        return {k: v for k, v in data.items() if v['split'] == self.split}

    def __len__(self):
        item = list(self.data.values())
        return len(item)

    def upload_image(self, image_list, images_type = "positive"):
        image_tensors = []
        if images_type == "negative":
            img_counter = 0
            for image_id in image_list:
                # image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
                try:
                    # image = Image.open(image_path).convert("RGB")
                    image  = load_image("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_HDF5",str(image_id))
                    image_tensors.append(image)
                    img_counter += 1
                    if img_counter == 23:
                        break
                except FileNotFoundError:
                    print(f"Image {image_id}.jpg not found in {self.image_folder}")
                    continue

            if img_counter != 23:
                while img_counter < 23:
                    for image_id in image_list:
                        #image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
                        try:
                            image  = load_image("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_HDF5",str(image_id))
                            image_tensors.append(image)
                            img_counter += 1
                            if img_counter == 23:
                                break

                        except FileNotFoundError:
                            print(f"Image {image_id}.jpg not found in {self.image_folder}")   
                            continue               
        else:                  
            img_counter = 0
            for image_id in image_list:
                #image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
                try:
                    image  = load_image("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_HDF5",str(image_id))
                    image_tensors.append(image)
                    img_counter += 1
                    if img_counter == 2:
                        break
                except FileNotFoundError:
                    print(f"Image {image_id}.jpg not found in {self.image_folder}") 
                    continue

            if img_counter != 2:
                while img_counter < 2:
                    for image_id in image_list:
                        # image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
                        try:
                            image  = load_image("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_HDF5",str(image_id))
                            image_tensors.append(image)
                            img_counter += 1
                            if img_counter == 2:
                                break

                        except FileNotFoundError:
                            print(f"Image {image_id}.jpg not found in {self.image_folder}")
                            continue 
        return image_tensors
    
    def __getitem__(self, idx):
        qid = list(self.data.keys())[idx]
        sample = list(self.data.values())[idx]
        positive_img_ids = sample.get('pos_imgs', [])
        negative_img_ids = sample.get('neg_imgs', [])
        #question_text = sample.get('question', '')

        # Loading the images from Dataset
        positive_image_encoding = self.upload_image(positive_img_ids)        
        negative_image_encoding = self.upload_image(negative_img_ids, images_type = "negative")
        
        # Tokeninzing the question text
        # question_encoding = self.bert_tokenizer(question_text, return_tensors='pt', padding='max_length', truncation=True, max_length=66)
        question_encoding = load_question("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/questions_HDF5",qid).to(self.device)
 
        # Processing images
        positive_image_encoding = [self.clip_preprocess(images=img, return_tensors='pt')['pixel_values'].squeeze(0) for img in positive_image_encoding]
        negative_image_encoding = [self.clip_preprocess(images=img, return_tensors='pt')['pixel_values'].squeeze(0) for img in negative_image_encoding]

        ##positive_image_encoding = [img.squeeze(0) for img in positive_image_encoding]
        ##negative_image_encoding = [img.squeeze(0) for img in negative_image_encoding]
            
        """
        final_dict = {
            "question_encoding": question_encoding,
            "positive_image_encoding": torch.stack(positive_image_encoding).squeeze(1),
            "negative_image_encoding": torch.stack(negative_image_encoding).squeeze(1)
        }"""

        
        positive_image_encoding = torch.stack(positive_image_encoding).squeeze(1)
        negative_image_encoding = torch.stack(negative_image_encoding).squeeze(1)

        return question_encoding, positive_image_encoding, negative_image_encoding