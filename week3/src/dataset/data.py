from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

import random
import torch


class Data(Dataset):
    def __init__(self, data, partition, img_path: str, chars: list, char2idx: dict, text_max_len: int = 201):
        self.data = data
        self.img_path = img_path
        self.partition = partition
        
        self.chars = chars
        self.char2idx = char2idx
        
        self.num_captions = 1
        self.max_len = text_max_len
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        
        # Image processing
        img_name = item["Image_Name"].reset_index(drop=True)[0]
        img = Image.open(f'{self.img_path}/{img_name}.jpg').convert('RGB')
        img = self.img_proc(img)
    
        # Caption processing
        caption = item["Title"].reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_list = list(caption)
        final_list = [self.chars[0]]
        final_list.extend(cap_list)
        final_list.extend([self.chars[1]])
        gap = self.max_len - len(final_list)
        final_list.extend([self.chars[2]]*gap)
        cap_idx = [self.char2idx[i] for i in final_list]
        return img, torch.tensor(cap_idx)