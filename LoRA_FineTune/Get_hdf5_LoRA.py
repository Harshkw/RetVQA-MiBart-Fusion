import h5py
import torch

def load_image(hdf5_file_path, dataset_name):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        img_np = hdf5_file[dataset_name]  # Access the dataset
        return img_np[:]
    
def load_question(hdf5_file_path, dataset_name):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        question_np = hdf5_file[dataset_name]  # Access the dataset
        question_np = torch.tensor(question_np[0])
        return question_np
    
def load_image_embed(hdf5_file_path, dataset_name):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        img_embed = hdf5_file[dataset_name]  # Access the dataset
        img_embed = torch.tensor(img_embed[0])
        return img_embed