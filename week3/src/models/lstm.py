import torch
import torch.nn as nn
import random

from transformers import ResNetModel


class Model(nn.Module):
    def __init__(self, 
                 num_char: int, 
                 char2idx: dict, 
                 text_max_len: int = 201, 
                 device: torch.device = None,
                 num_layers: int = 1,
                 freeze_backbone: bool = True,
                 dropout_rate: float = 0.3,
                 embedding_dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()
        
        # Load ResNet and define layers
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.device)
        
        # Feature normalization and dropout
        self.feature_norm = nn.LayerNorm(512)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # LSTM with dropout
        self.LSTM = nn.LSTM(512, 512, num_layers=num_layers, dropout=dropout_rate, batch_first=False)
        
        # Embed with dropout
        self.embed = nn.Embedding(num_char, 512)
        self.embed_dropout = nn.Dropout(embedding_dropout)
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Linear(512 * 2, 1)
        
        # Project to num_char
        self.pre_proj_dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(512, num_char)
        
        # Freeze ResNet
        if freeze_backbone:
            for param in self.resnet.parameters():
                print(f"Freezing {param.shape}")
                param.requires_grad = False
        
        # Some parameters
        self.num_layers = num_layers
        self.text_max_len = text_max_len
        self.char2idx = char2idx
        self.dropout_rate = dropout_rate
        
    def attend_features(self, hidden, feature_map):
        """
        Apply attention mechanism to focus on relevant image features
        
        """
        # Reshape and repeat hidden state
        hidden_expanded = hidden[-1].unsqueeze(1).repeat(1, feature_map.size(1), 1)  # [batch, seq_len, hidden_dim]
        combined = torch.cat([feature_map, hidden_expanded], dim=2)  # [batch, seq_len, hidden_dim*2]
        
        # Calculate attention scores
        attn_scores = self.attention(combined).squeeze(2)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # [batch, seq_len, 1]
        
        # Apply attention weights to features
        context = torch.sum(feature_map * attn_weights, dim=1).unsqueeze(0)  # [1, batch, hidden_dim]
        return context

    def forward(self, img, target = None, teacher_forcing_ratio: float=0.5):
        print(target.shape) if target is not None else print("No target")
        batch_size = img.shape[0]
        
        # Extract features with ResNet
        resnet_output = self.resnet(img)
        feat = resnet_output.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        # Apply feature normalization and dropout
        feat = self.feature_norm(feat)
        feat = self.feature_dropout(feat) if self.training else feat
        
        # Initialize hidden and cell states
        hidden = feat.repeat(self.num_layers, 1, 1) # num_layers, batch, 512
        cell_state = torch.zeros_like(hidden) # num_layers, batch, 512
        
        # Start token embedding
        start = torch.tensor(self.char2idx['<SOS>']).to(self.device)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        # Apply embedding dropout
        inp = self.embed_dropout(start_embeds) if self.training else start_embeds
        
        outputs = []
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio and self.training else False
        
        # Get last hidden layer output from ResNet
        if self.use_attention:
            feature_map = resnet_output.last_hidden_state  # [batch, seq_len, hidden_dim]
            batch_size, channels, height, width = feature_map.shape
            feature_map = feature_map.permute(0, 2, 3, 1)
            feature_map = feature_map.reshape(batch_size, height*width, channels)
        
        # Loop through each sequence
        for t in range(self.text_max_len): # rm <SOS>
            # Apply attention if enabled
            if self.use_attention:
                context = self.attend_features(hidden, feature_map)
                hidden = hidden + context
            
            # Get output and hidden states from LSTM
            out, (hidden, cell_state) = self.LSTM(inp, (hidden, cell_state))
            out = self.pre_proj_dropout(out) if self.training else out
            
            # Project to vocabulary size
            projection = self.proj(out[-1])  # [batch, num_char]
            outputs.append(projection)
            
            # Prepare next input based on teacher forcing or prediction
            if use_teacher_forcing and target is not None and t < target.size(1) - 1:
                # Teacher forcing: use ground truth as next input
                word_idx = target[:, t].to(self.device)
            else:
                # Without teacher forcing: use own prediction
                word_idx = projection.argmax(dim=1)
            
            # Embed next input
            word_embedding = self.embed(word_idx).unsqueeze(0)  # [1, batch, 512]
            inp = self.embed_dropout(word_embedding) if self.training else word_embedding
    
        # Stack outputs and reshape
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, num_char]
        outputs = outputs.permute(0, 2, 1)  # [batch, num_char, seq_len]
        return outputs
    