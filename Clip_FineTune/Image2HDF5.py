import h5py
import PIL
from PIL import Image
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertModel
from Extended_BertModel import BertWithLinear
from transformers import CLIPProcessor
from transformers import CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def create_hdf5_from_images(image_folder, json_path, hdf5_file_path_images, hdf5_file_path_questions, hdf5_file_path_images_embed):
    image_files = os.listdir(image_folder)
    """
    image_files = os.listdir(image_folder)
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model1 = BertWithLinear(bert_model).to(device)
    bert_model_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(json_path, 'r') as f:
            data = json.load(f)

    with h5py.File(hdf5_file_path_questions, 'w') as hdf5_file:
        for qid, item in tqdm(data.items()):
            question = item['question']
            question_encoding = bert_model_tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=66).to(device)
            question_embeddings = bert_model1(question_encoding['input_ids'].to(device), question_encoding['attention_mask'].to(device))[1].cpu().detach().numpy()

            dataset_name = f'{qid}'
            hdf5_file.create_dataset(dataset_name, data=question_embeddings)

    """
    
    # Open an HDF5 file
    """
    with h5py.File(hdf5_file_path_images, 'w') as hdf5_file:
        count = 0
        for image_file in tqdm(image_files):
            
            idx = int(image_file.split('.')[0])
            image_path = os.path.join(image_folder, image_file)
            try:
                img_np = Image.open(image_path).convert("RGB")
                dataset_name = f'{idx}'
                hdf5_file.create_dataset(dataset_name, data=img_np)

            except (PIL.UnidentifiedImageError, OSError) as e:
                count += 1
                image_path = os.path.join(image_folder, f"{count}.jpg")
                img_np = Image.open(image_path).convert("RGB")
                dataset_name = f'{idx}'
                print(idx)
                hdf5_file.create_dataset(dataset_name, data=img_np)
                #print(f"Skipping corrupted file {image_file}: {e}")
                continue

        print(f"Skipped {count} corrupted files")
        """
    
    with h5py.File(hdf5_file_path_images_embed, 'w') as hdf5_file:
        count = 0
        clip_model = CLIPModel.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clip_11.pth").to(device)
        clip_preprocess = CLIPProcessor.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clipProcessor_11.pth")
        j = 0
        for image_file in tqdm(image_files):

            idx = int(image_file.split('.')[0])
            image_path = os.path.join(image_folder, image_file)
            try:
                img_np = Image.open(image_path).convert("RGB")
                dataset_name = f'{idx}'
                img_pre = clip_preprocess(images=img_np, return_tensors='pt')['pixel_values']
                image_embedings = clip_model.get_image_features(pixel_values=img_pre.to(device)).cpu().detach().numpy()
                hdf5_file.create_dataset(dataset_name, data=image_embedings)

            except (PIL.UnidentifiedImageError, OSError) as e:
                count += 1
                image_path = os.path.join(image_folder, f"{count}.jpg")
                img_np = Image.open(image_path).convert("RGB")
                dataset_name = f'{idx}'
                print(idx)
                img_pre = clip_preprocess(images=img_np, return_tensors='pt')['pixel_values']
                image_embedings = clip_model.get_image_features(pixel_values=img_pre.to(device)).cpu().detach().numpy()
                hdf5_file.create_dataset(dataset_name, data=image_embedings)
                #print(f"Skipping corrupted file {image_file}: {e}")
                continue
            j += 1
            if j == 10:
                break

        print(f"Skipped {count} corrupted files")

# Create the HDF5 file from a folder of images
create_hdf5_from_images('/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/images/VG_100K', 
                        '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/retvqa_release_v1.json',
                        '/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_HDF5',
                        '/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/questions_HDF5',
                        '/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_embed_HDF5')
