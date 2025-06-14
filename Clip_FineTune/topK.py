import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
from tqdm import tqdm
import os
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel
from transformers import CLIPModel
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from CustomDataset import DatasetPreparation
from Extended_BertModel import BertWithLinear
from LossFunction import ContrastiveLoss
from transformers import BertTokenizer
from transformers import CLIPProcessor
# Load the trained model and processor
model = CLIPModel.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clip_24.pth")
processor = CLIPProcessor.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clipProcessor_24.pth")  # or your custom processor
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model1 = BertWithLinear(bert_model)
# Set model to evaluation mode
model.eval()
# Load tokenizer for questions
from transformers import BertTokenizer
from transformers import BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

json_file = '/home/monu_harsh/Monu/dataset/retvqa_release_v1.json'
# Load the data from JSON file
with open(json_file, "r") as file:
    data = json.load(file)

output_dict = {}
count = 0
correct = 0
total = 0
i=0
for qid, item in tqdm(data.items()):
    print(i)
    i+=1
    if i>10:
         break
    question = item['question']
    pos_img_ids = item['pos_imgs']
    neg_img_ids = item['neg_imgs']
    
    # Combine positive and negative image IDs
    img_ids = pos_img_ids + neg_img_ids

    # Tokenize question
    question_encoding = bert_model_tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=77)

    # Store image similarity scores
    img_scores = []
    images = '/home/monu_harsh/Monu/dataset/images'  
    for img_id in img_ids:
        # Load image
        img_path = os.path.join(images, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Preprocess image
        image_encoding = processor(images=image, return_tensors="pt")

        # Compute embeddings
        with torch.no_grad():
            question_embeddings = model.get_text_features(**question_encoding)
            img_embeddings = model.get_image_features(**image_encoding)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(question_embeddings, img_embeddings)
        img_scores.append((img_id, similarity.item()))

    # Sort images by similarity scores in descending order and pick top 2
    img_scores.sort(key=lambda x: x[1], reverse=True)
    top_2_imgs = [img_scores[0][0], img_scores[1][0]]
    common_elements = set(top_2_imgs).intersection(pos_img_ids)
    if len(common_elements)>0:
            count+=1
    correct+=len(common_elements)
    total+=2
    # Store the top 2 images for this question
    output_dict[qid] = top_2_imgs

# Convert to desired format
output = [{qid: top_2_imgs} for qid, top_2_imgs in output_dict.items()]
with open('/home/monu_harsh/results/val_scores.txt', 'a') as file:
        file.write(f"Total correct image percentage in all =  {100*correct/total}\n")
with open('/home/monu_harsh/results/val_scores.txt', 'a') as file:
    file.write(f"Atleast one Correct per Question in all =  {200*count/total}\n")
# Save the results in a new JSON file
with open("/home/monu_harsh/Monu/dataset/retvqa_release_v1_finetuned.json", "w") as outfile:
    json.dump(output, outfile, indent=4)



