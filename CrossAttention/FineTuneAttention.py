import os
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel
from transformers import CLIPModel
from torch.optim import Adam
from tqdm import tqdm
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from CustomData import CustomDataset
from Extended_BertModel import BertWithLinear
from Loss import AttentionLoss
from Loss import CustomLoss
from CrossAttention import CrossAttention
from transformers import BertTokenizer
from transformers import CLIPProcessor
from Get_hdf5 import load_image
from Get_hdf5 import load_question
# Usage
#total_train_ids = 8
#total_test_ids = 8
batch_size = 256
#test_batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

images_folder = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/images/VG_100K'
json_file = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/retvqa_release_v1.json'
val_json_file = '/home/monu_harsh/Monu/dataset/retvqa_val_v1.json'

# Prepare Train Dataset
train_dataset_prep = CustomDataset(
    json_file= json_file,
    image_folder= images_folder,
    device=device,
    #num_images = total_train_ids,
    split = "train"
)
val_dataset_prep = CustomDataset(
    json_file= json_file,
    image_folder= images_folder,
    device=device,
    #num_images = total_test_ids,
    split = "val"
)

print("Creating Train Dataset and preprocessing images/text.....................................................")
print("Loading Train Dataset....................................................................................")
train_dataloader = DataLoader(train_dataset_prep, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset_prep, batch_size=batch_size, shuffle=False)
print("Train Dataloader Ready")
# Instantiate the CrossAttention layer and AttentionLoss
query_dim = 512  # question embedding size
key_value_dim = 512  # image embedding size
num_positive = 2
num_negative = 23

cross_attention = CrossAttention(query_dim, key_value_dim).to(device)
attention_loss = AttentionLoss()
custom_loss = CustomLoss()
optimizer = optim.AdamW(cross_attention.parameters(), lr=0.001)


model_save_path = "/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/Model_attention/cross_attention_newloss_1st{epoch}.pth"
# Fine Tune
num_epochs = 40
for epoch in range(num_epochs):
    b = 0
    for question_encoding, pos_image_embeddings, neg_image_embeddings,total_ids,qid in tqdm(train_dataloader):
        optimizer.zero_grad()
        # Perform cross-attention for each question and its corresponding images
        loss = 0
        for i in range(len(question_encoding)):
            # Single question embedding and its corresponding images
            q_emb = question_encoding[i].unsqueeze(0)  # Shape [1, 512]
            ####
            q_emb_next = question_encoding[(i+1)%len(question_encoding)].unsqueeze(0)  # Shape [1, 512]
            ####

            img_embs = torch.cat((pos_image_embeddings[i][0:2], neg_image_embeddings[i][0:23]), dim=0)  # Shape [25, 512]

            # Apply cross-attention
            output, attn_weights = cross_attention(q_emb, img_embs)
        
            # Calculate loss based on attention weights
            loss += attention_loss(attn_weights, num_positive, num_negative)
            # loss += custom_loss(q_emb, q_emb_next, pos_image_embeddings[i])

        loss.backward()
        optimizer.step()   
        if b%20==0:
            with open('/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/results_attention/train_newloss_batch_2nd.txt', 'a') as file2:
                file2.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss batch {b}: {loss.item()}" + "\n")
        b+=1

    torch.save(cross_attention.state_dict(), model_save_path.format(epoch=epoch+1)) 

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}")
    with open('/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/results_attention/train_newloss_2nd.txt', 'a') as file1:
        file1.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss Final: {loss.item()}" + "\n")
    
    

    ################################################################################################# 
    #For all validation data at once
    correct=0
    total=0
    count1=0
    count2=0
    for question_encoding, pos_image_embeddings, neg_image_embeddings, total_ids,qid in tqdm(val_dataloader):
        for i in range(len(question_encoding)):
            # Single question embedding and its corresponding images
            q_emb = question_encoding[i].unsqueeze(0)  # Shape [1, 512]
            img_embs = torch.cat((pos_image_embeddings[i][0:2], neg_image_embeddings[i][0:23]), dim=0)  # Shape [25, 512]

            # Apply cross-attention
            output, attn_weights = cross_attention(q_emb, img_embs)
            # Sort images by attention weights and retrieve top 2
            topk_indices = torch.topk(attn_weights[0], 2).indices  # Get indices of top 2 images

            # Check if the retrieved images are among the positive images
            positive_img_indices = set(range(num_positive))  # Indices for positive images
            retrieved_indices = set(topk_indices.cpu().tolist())  # Convert to list for comparison
            comm_ele = retrieved_indices.intersection(positive_img_indices)
            correct += len(comm_ele)  # Check if top 2 include positive images
            total += 2
            if len(comm_ele)>0:
                count1+=1
            if len(comm_ele)>1:
                count2+=1

    accuracy = (correct / total) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")

    #################################################################################################
    #################################################################################################
    with open('/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/results_attention/val_newloss_scores_2nd.txt', 'a') as file3:
        file3.write(f"Epoch {epoch+1}: {100*correct/total}\n")
        file3.write(f"Atleast one Correct per Question in Epoch {epoch+1}: {200*count1/total}\n")
        file3.write(f"Atleast two Correct per Question in Epoch {epoch+1}: {200*count2/total}\n")
    
        
    