import torch
import torch.nn as nn

from transformers import ResNetModel


class Model(nn.Module):
    def __init__(self, 
                 num_char: int, 
                 char2idx: dict, 
                 text_max_len: int = 201, 
                 device: torch.device = None,
                 num_layers: int = 1,
                 freeze_backbone: bool = True):
        super().__init__()
        # Load ResNet and define layers
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.device)
        self.LSTM = nn.LSTM(512, 512, num_layers=num_layers)
        self.proj = nn.Linear(512, num_char)
        self.embed = nn.Embedding(num_char, 512)
        
        # Freeze ResNet
        if freeze_backbone:
            for param in self.resnet.parameters():
                print(f"Freezing {param.shape}")
                param.requires_grad = False
        
        # Some parameters
        self.num_layers = num_layers
        self.text_max_len = text_max_len
        self.char2idx = char2idx

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        
        # Initialize hidden and cell states
        hidden = feat.repeat(self.num_layers, 1, 1) # num_layers, batch, 512
        cell_state = torch.zeros_like(hidden) # num_layers, batch, 512
        
        # Start token embedding
        start = torch.tensor(self.char2idx['<SOS>']).to(self.device)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        
        # Loop through LSTM
        for t in range(self.text_max_len - 1): # rm <SOS>
            out, (hidden, cell_state) = self.LSTM(inp, (hidden, cell_state))
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512
    
        # Project to num_char
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80
        res = res.permute(0, 2, 1) # batch, 80, seq
        return res
    