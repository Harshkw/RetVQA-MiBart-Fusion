import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, BertTokenizer, BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType  # Install `peft` library
from CustomDataset_LoRA import DatasetPreparation
from LossFunction import ContrastiveLoss
from Get_hdf5_LoRA import load_image
from Get_hdf5_LoRA import load_question
import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
images_folder = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/images/VG_100K'
json_file = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/retvqa_release_v1.json'
val_json_file = '/home/monu_harsh/Monu/dataset/retvqa_val_v1.json'

# Hyperparameters
batch_size = 256
num_epochs = 35
learning_rate = 1e-3

# Load CLIP model and processor
print("Initializing CLIP model and processor...")
clip_model = CLIPModel.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/LoRA_FineTune/LoRA_FineTuned_Models/clip_lora_epoch_4").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze all parameters
for param in clip_model.parameters():
    param.requires_grad = False

for name, _ in clip_model.named_modules():
    print(name)

# LoRA configuration
print("Configuring LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # Specify task type as vision
    r=8,  # Low-rank adaptation dimension
    lora_alpha=16,  # Scaling factor
    #target_modules=["vision_model.encoder.layers.11.self_attn.k_proj", "vision_model.encoder.layers.11.self_attn.q_proj", "vision_model.encoder.layers.11.self_attn.v_proj"],  # Specify target modules in CLIP
    target_modules=["visual_projection"],  # Specify target modules in CLIP
    lora_dropout=0.1,
    bias="none"
)

# Apply LoRA to the CLIP model
clip_model = get_peft_model(clip_model, lora_config)
clip_model.print_trainable_parameters()

# Prepare tokenizer and dataset
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = DatasetPreparation(
    json_file=json_file,
    image_folder=images_folder,
    clip_preprocess=clip_processor,
    bert_tokenizer=bert_tokenizer,
    device=device,
    #num_images=128,
    split="train"
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

with open(val_json_file, "r") as file:
    data_val = json.load(file)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(clip_model.parameters(), lr=learning_rate)
criterion = ContrastiveLoss()

# Fine-tuning loop
print("Starting fine-tuning with LoRA...")
for epoch in range(4, num_epochs):
    clip_model.train()
    epoch_loss = 0
    for question_encoding, positive_img_encodings, negative_img_encodings in tqdm(train_dataloader):
        optimizer.zero_grad()

        # Forward pass
        pos_image_embeddings = [clip_model.get_image_features(pixel_values=img.squeeze(1).to(device)) for img in positive_img_encodings]
        neg_image_embeddings = [clip_model.get_image_features(pixel_values=img.squeeze(1).to(device)) for img in negative_img_encodings]

        pos_image_embeddings = torch.cat(pos_image_embeddings)
        neg_image_embeddings = torch.cat(neg_image_embeddings)

        question_embeddings_pos = question_encoding.repeat(2, 1)
        question_embeddings_neg = question_encoding.repeat(23, 1)

        loss = criterion(question_embeddings_pos, question_embeddings_neg, pos_image_embeddings, neg_image_embeddings, batch_size)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save LoRA-adapted model
    clip_model.save_pretrained(f"/home/monu_harsh/Harshwardhan/mi_bart/LoRA_FineTune/LoRA_FineTuned_Models/clip_lora_epoch_{epoch + 1}")

    ################################################################################################# 
    #For all validation data at once
    correct=0
    count = 0
    total=0
    i = 0
    for qid, item in tqdm(data_val.items()):
        question = item['question']
        real_pos = item['pos_imgs']
        img_ids = item['pos_imgs'] + item['neg_imgs']

        val_images_all = []
        for ids in img_ids:
            val_images_all.append(load_image("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/images_HDF5",str(ids)))
        image_encodings_val = clip_processor(images=val_images_all, return_tensors="pt").to(device)
        repeat  = len(val_images_all)

        # Compute embeddings
        with torch.no_grad():
            img_embeddings_val = clip_model.get_image_features(**image_encodings_val)

        que_emb = load_question("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/RetVQA_Data_HDF5/questions_HDF5",str(qid)).to(device)
        que_emb = que_emb.repeat(repeat,1)
        similarities = torch.nn.functional.cosine_similarity(que_emb, img_embeddings_val, dim=-1)
        
        simWithIndex=[]
        similarities = similarities.tolist()
        for j in range(0,len(similarities)):
            simWithIndex.append([similarities[j],j])
        # Get top 2 images with highest similarity scores
        simWithIndex.sort(key=lambda x: x[0], reverse=True)
        ind1 = simWithIndex[0][1]
        ind2 = simWithIndex[1][1]
        top_2_imgs = [img_ids[ind1],img_ids[ind2]]        
        # print(top_2_imgs)
        # print(real_pos)
        common_elements = set(top_2_imgs).intersection(real_pos)
        if len(common_elements)>0:
            count+=1
        correct+=len(common_elements)
        total+=2
        # print(common_elements)

    #################################################################################################
    #################################################################################################
    with open('/home/monu_harsh/Harshwardhan/mi_bart/LoRA_FineTune/LoRA_Results/val_scores_LoRA.txt', 'a') as file:
        file.write(f"Epoch {epoch+1}: {100*correct/total}\n")
        file.write(f"Atleast one Correct per Question in Epoch {epoch+1}: {200*count/total}\n")
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation accuracy: {100*correct/total}" + "\n")

# Save final model
clip_model.save_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/LoRA_FineTune/LoRA_FineTuned_Models/clip_lora_final")
print("Fine-tuning complete. Model saved.")
