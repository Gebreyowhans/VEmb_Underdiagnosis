from open_clip import create_model_from_pretrained
import os
import pandas as pd
import numpy as np
import sys
import torch
from PIL import Image
from pickle import dump, load

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir)))


sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname('__file__'), os.pardir)))

PATH_TO_IMAGES = '/datasets/chexpert/'

df_path = "/h/gebrehb/gebrehb_link/ClipEmbedding/dataframes/chexpert_df.csv"

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

# Paths
# Root directory of the project
root_dir = os.path.dirname(os.path.dirname('__file__'))

metas = ["race-bin", "race-cls", "gender",
         "age_decile", "insurance", "disease"]

# mimic_jpg_folder = os.path.join(root_dir, "/datasets", "chexpert")
mimic_jpg_folder = os.path.join("/datasets", "chexpert")

biomedclip_embedding_folder = os.path.join(
    root_dir, "data", "biomedclip-embedding")
biomedclip_embedding_path = os.path.join(
    biomedclip_embedding_folder, "embedding_from_biomedclip.pkl")
train_test_idx_path_biomedclip = os.path.join(
    biomedclip_embedding_folder, "train_test_idx.pkl")

extracted_embedding_folder = os.path.join(
    root_dir, "data", "generalized-image-embedding")
train_test_idx_path = os.path.join(
    extracted_embedding_folder, "train_test_idx.pkl")

os.makedirs(biomedclip_embedding_folder, exist_ok=True)


# Function to remove correlated features

def get_embedding_from_biomedclip_and_metadata(df, r_dir, model,
                                               preprocess, device):
    print("Extract targets and embedding with BiomedCLIP...")
    target = dict()

    # Extract embedding data
    # Normalize the path for consistency across different operating systems
    df['Path'] = df['Path'].apply(lambda p: os.path.normpath(p))

    embeddings = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows()):
        image_path = os.path.join("/datasets", "chexpert", row['Path'])

        # print(f'image path : {image_path}')

        # image_path = os.path.join(
        # r_dir, "datasets", "chexpert", row['Path'])

        try:
            # print(f"Compute embedding biomedclip for: {image_path}")
            image = preprocess(Image.open(image_path).convert(
                'RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            embeddings.append(embedding.cpu().numpy())
            valid_indices.append(idx)
        except FileNotFoundError:
            print(f"Image {image_path} not found.")
            embeddings.append(np.zeros((1, 512)))
        except OSError as e:
            print(f"Error processing image {image_path}: {e}")
            embeddings.append(np.zeros((1, 512)))

    embedding_data = np.vstack(embeddings)
    patient_id = np.array(list(df["Path"].values)).reshape(-1, 1)
    return embedding_data, target, valid_indices, patient_id


# Select patients for training
print(f'Select patients...')
data_df = pd.read_csv(df_path)

# data_df.drop_duplicates(subset='subject_id', inplace=True)
# data_df = data_df.reset_index()

# get unique subject ids
unique_ids = np.array(
    list(data_df["Path"].unique())).reshape(-1, 1)


print("Load BiomedCLIP...")
# Load BiomedCLIP
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# this function is valid for both Pubmed and Biomed
X, y, valid_id, ids = get_embedding_from_biomedclip_and_metadata(data_df, root_dir, model,
                                                                 preprocess, device)

print('Saving embedding and target variables in raw format...')
with open(biomedclip_embedding_path, 'wb') as f:
    dump((X, y, valid_id, ids), f)
