import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from CrossAttention import CrossAttention
from CustomData import CustomDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
path = "/home/monu_harsh/Harshwardhan/mi_bart/CrossAttention/Model_attention/cross_attention3.pth"
query_dim = 512  # question embedding size
key_value_dim = 512  # image embedding size

cross_attention = CrossAttention(query_dim, key_value_dim).to(device)
cross_attention.load_state_dict(torch.load(path))
cross_attention.eval()
batch_size = 256

images_folder = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/images/VG_100K'
json_file = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/retvqa_release_v1.json'

total_dataset_prep = CustomDataset(
    json_file= json_file,
    image_folder= images_folder,
    device=device,
    #num_images = total_test_ids,
    split = "total"
)

total_dataloader = DataLoader(total_dataset_prep, batch_size=batch_size, shuffle=False)
#For all validation data at once
correct=0
total=0
count1=0
count2=0
output_dict = {}
for question_encoding, pos_image_embeddings, neg_image_embeddings, total_ids,qid in tqdm(total_dataloader):
    for i in range(len(question_encoding)):
        # Single question embedding and its corresponding images
        q_emb = question_encoding[i].unsqueeze(0)  # Shape [1, 512]
        img_embs = torch.cat((pos_image_embeddings[i][0:2], neg_image_embeddings[i][0:23]), dim=0)  # Shape [25, 512]

        # Apply cross-attention
        output, attn_weights = cross_attention(q_emb, img_embs)
        # Sort images by attention weights and retrieve top 2
        topk_indices = torch.topk(attn_weights[0], 2).indices  # Get indices of top 2 images

        # Check if the retrieved images are among the positive images
        positive_img_indices = set(range(2))  # Indices for positive images
        retrieved_indices = set(topk_indices.cpu().tolist())  # Convert to list for comparison
        comm_ele = retrieved_indices.intersection(positive_img_indices)

        top_2_imgs = [total_ids[i][idx].item() for idx in retrieved_indices]
        # Store the top 2 images for this question
        output_dict[str(qid[i])] = top_2_imgs
        correct += len(comm_ele)  # Check if top 2 include positive images
        total += 2
        if len(comm_ele)>0:
            count1+=1
        if len(comm_ele)>1:
            count2+=1

# Convert to desired format
output = [{qid: top_2_imgs} for qid, top_2_imgs in output_dict.items()]
accuracy = (correct / total) * 100
print(f"Validation Accuracy: {accuracy:.2f}%")
with open('/home/monu_harsh/results/val_scores_att_test.txt', 'a') as file:
        file.write(f"Total correct image percentage in all =  {100*correct/total}\n")
with open('/home/monu_harsh/results/val_scores_att_test.txt', 'a') as file:
    file.write(f"Atleast one Correct per Question in all =  {200*count1/total}\n")
# Save the results in a new JSON file
with open("/home/monu_harsh/Monu/dataset/retrieved_data_pos_all_finetune_attention.json", "w") as outfile:
    json.dump(output, outfile, indent=4)
    

