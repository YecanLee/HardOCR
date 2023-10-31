from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from PIL import Image
import torch
import pandas as pd

# Define the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file = 'C:/Users/ra78lof/occinference/byte-level-BPE.tokenizer.json')

# Add PAD token to the vocabulary, otherwise it will throw an error
tokenizer.add_special_tokens({'pad_token': "pad_token"})

class CustomDataset(Dataset):
    def __init__(self, excel_file, img_dir, tokenizer = tokenizer, feature_extractor = None, transform=None, max_target_length = 45):
        self.data = pd.read_excel(excel_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data['ImageName'][idx]
        text = self.data['Labels'][idx]
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        # The labels MUST be tokenized and padded
        labels = self.tokenizer(text, padding = 'max_length', max_length = self.max_target_length).input_ids
        

        return image, torch.as_tensor(labels)