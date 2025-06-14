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
from CustomDataset import DatasetPreparation
from Extended_BertModel import BertWithLinear
from LossFunction import ContrastiveLoss
from transformers import BertTokenizer
from transformers import CLIPProcessor
from Get import load_image
from Get import load_question
# Usage
#total_train_ids = 8
#total_test_ids = 128
batch_size = 256
#test_batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

images_folder = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/images/VG_100K'
json_file = '/home/monu_harsh/Harshwardhan/mi_bart/RetVQA_Dataset/retvqa_release_v1.json'
val_json_file = '/home/monu_harsh/Monu/dataset/retvqa_val_v1.json'


# Initialize models
print("Initializing Clip and Bert model...............................................")
clip_model = CLIPModel.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clip_24.pth").to(device)
clip_preprocess = CLIPProcessor.from_pretrained("/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clipProcessor_24.pth")
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model1 = BertWithLinear(bert_model).to(device)
bert_model_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare Train Dataset
train_dataset_prep = DatasetPreparation(
    json_file= json_file,
    image_folder= images_folder,
    clip_preprocess = clip_preprocess,
    bert_tokenizer = bert_model_tokenizer,
    device=device,
    #num_images = total_train_ids,
    split = "train"
)

print("Creating Train Dataset and preprocessing images/text.....................................................")
print("Loading Train Dataset....................................................................................")
train_dataloader = DataLoader(train_dataset_prep, batch_size=batch_size, shuffle=True)
print("Train Dataloader Ready")

""""
# Create Validation Dataset
val_dataset_prep = DatasetPreparation(
    json_file= json_file,
    image_folder= images_folder,
    clip_preprocess = clip_preprocess,
    bert_tokenizer = bert_model_tokenizer,
    device=device,
    num_images = total_test_ids,
    split = "val"
)

print("Creating Validation Dataset and preprocessing images/text......................................................")
print("Loading Validation Dataset.....................................................................................")
val_dataloader = DataLoader(val_dataset_prep, batch_size=test_batch_size, shuffle= True)
print("Validation Dataloader Ready")
"""

# Freeze all layers except the last layer
for name, param in clip_model.named_parameters():
    if 'vision_model' in name and 'layernorm' not in name:
        param.requires_grad = False

# Unfreeze the last layer
for param in clip_model.vision_model.encoder.layers[-1].parameters():
    param.requires_grad = True

# Define optimizer and loss function
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=5e-5)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=1e-5)
criterion = ContrastiveLoss()
#test_loss_fn = ContrastiveLoss()


#########################################
projection_layer = nn.Linear(768, 512).to(device)
# Load the data from JSON file
with open(val_json_file, "r") as file:
    data_val = json.load(file)
#########################################

print("Loading Validation Dataset...............................................................................")
"""
val_images_all=[]
val_image_ind_all = []
real_pos_all=[]
que_all = []
for qid, item in tqdm(data_val.items()):
    question = item['question']
    real_pos = item['pos_imgs']
    img_ids = item['pos_imgs'] + item['neg_imgs']

    # Tokenize question
    question_encoding_val = bert_model_tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=66).to(device)

    # Load and preprocess all images at once
    images_val = []
    img_ind = []
    for img_id in img_ids:
        img_path = os.path.join(images_folder, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        images_val.append(image)
        img_ind.append(img_id)
    val_images_all.append(images_val)
    real_pos_all.append(real_pos)
    val_image_ind_all.append(img_ind)
    with torch.no_grad():
        question_embeddings_val = bert_model(**question_encoding_val).last_hidden_state.mean(dim=1)
        question_embeddings_val = projection_layer(question_embeddings_val)
    que_all.append(question_embeddings_val)
"""
# Fine Tune
num_epochs = 35

for epoch in range(24,num_epochs):
    b = 0
    for question_encoding, positive_img_encodings, negative_img_encodings in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        # Getting input_ids from tokenized questions
        #question_inputs = {k: v.squeeze(1) for k, v in question_encoding.items()}

        #input_ids = question_inputs["input_ids"]
        #input_ids = input_ids.squeeze(1)
        #attention_masks = question_inputs["attention_mask"]
        #attention_masks = attention_masks.squeeze(1)

        # Passing thorugh the bert model      
        #outputs = bert_model1(input_ids = input_ids.to(device), attention_mask = attention_masks.to(device))
        #question_embedding = outputs[1]
        # Shape of question_embedding [batch_size, 512]

        # Getting the embedding for the images after processing
        pos_image_embeddings = [clip_model.get_image_features(pixel_values=img.squeeze(1).to(device)) for img in positive_img_encodings]
        neg_image_embeddings = [clip_model.get_image_features(pixel_values=img.squeeze(1).to(device)) for img in negative_img_encodings]


        pos_image_embeddings = torch.cat(pos_image_embeddings)    # Shape [batch_size, 2, 512]
        neg_image_embeddings = torch.cat(neg_image_embeddings)    # Shape [batch_size, 9, 512]
        question_embeddings_pos = question_encoding.repeat(2,1)
        question_embeddings_neg = question_encoding.repeat(23,1)
        

        loss = criterion(question_embeddings_pos, question_embeddings_neg, pos_image_embeddings, neg_image_embeddings, batch_size)
        loss.backward()
        optimizer.step()
        with open('/home/monu_harsh/results/train_loss_batch.txt', 'a') as file2:
            file2.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss batch {b}: {loss.item()}" + "\n")
        b+=1    

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}")
    with open('/home/monu_harsh/results/train_loss.txt', 'a') as file1:
        file1.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss Final: {loss.item()}" + "\n")
    
    if epoch != num_epochs-1:
        clip_model.save_pretrained(f'/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clip_{epoch+1}.pth')
        clip_preprocess.save_pretrained(f'/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Epochs/clipProcessor_{epoch+1}.pth')

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
        image_encodings_val = clip_preprocess(images=val_images_all, return_tensors="pt").to(device)
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
    with open('/home/monu_harsh/results/val_scores.txt', 'a') as file:
        file.write(f"Epoch {epoch+1}: {100*correct/total}\n")
    with open('/home/monu_harsh/results/val_scores.txt', 'a') as file:
        file.write(f"Atleast one Correct per Question in Epoch {epoch+1}: {200*count/total}\n")
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation accuracy: {100*correct/total}" + "\n")
    
# Save the model
clip_model.save_pretrained('/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Final_Epoch/clip_final.pth')
clip_preprocess.save_pretrained('/home/monu_harsh/Harshwardhan/mi_bart/Clip_FineTune/FineTunedModels/Final_Epoch/clipProcessor_final.pth')
print("Fine-tuning complete.")
print("Saving the final Model")
