import torch
import numpy as np

from src.metrics.metrics import Metric
from tqdm import tqdm


class Trainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.nn.Module, criterion: torch.nn.Module, tokenizer, device: torch.device = None, **kwargs):
        """Initializes the Trainer object.

        Args:
            model (torch.nn.Module): Model to train.
            optimizer (torch.nn.Module): Optimizer to use during training.
            criterion (torch.nn.Module): Loss function to use during training.
            device (torch.device): Device to use during training.
        """
        # Store the model, optimizer, and loss function
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = Metric()
        self.tokenizer = tokenizer
        
        # Store additional keyword arguments
        self.kwargs = kwargs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
        """Trains the model using the provided data loader.

        Args:
            train_loader (torch.utils.data.DataLoader): Data loader for training.
        """
        # Training loop
        for epoch in range(self.kwargs.get('epochs', 1)):
            # Train for one epoch
            loss, res = self.__train_one_epoch(train_loader)
            print(f'Train loss: {loss:.2f}, Metric: {res:.2f}, Epoch: {epoch + 1}')

            # Validate after every epoch
            loss_v, res_v = self.__eval_epoch(valid_loader)
            print(f'Validation loss: {loss_v:.2f}, Metric: {res_v:.2f}')

        # After training, evaluate on the test set
        loss_t, res_t = self.__eval_epoch(test_loader)
        print(f'Test loss: {loss_t:.2f}, Metric: {res_t:.2f}')
    
    def __train_one_epoch(self, train_loader: torch.utils.data.DataLoader):
        """Trains the model for one epoch using the provided data loader.

        Args:
            train_loader (torch.utils.data.DataLoader): Data loader for training.
        """
        self.model.train()  # Ensure the model is in training mode
        total_loss = 0
        total_metrics = 0
        num_batches = len(train_loader)

        for img, caption in tqdm(train_loader, desc='Training', total=num_batches):
            img, caption = img.to(self.device), caption.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(img)

            # Compute the loss
            loss = self.criterion(pred, caption)
            loss.backward()

            # Optimize the weights
            self.optimizer.step()

            # Compute the metrics (e.g., BLEU, ROUGE, METEOR)
            pred_text = torch.argmax(pred, dim=1)
            pred_text = pred_text.cpu().numpy()
            caption_text = caption.cpu().numpy()
            
            # Decode the token indices to text
            pred_decoded = self.tokenizer.decode(pred_text)
            caption_decoded = self.tokenizer.decode(caption_text)
            
            print(pred_decoded, caption_decoded)

            # Compute the metrics
            caption_decoded = np.array([[text] for text in caption_decoded], dtype=object)
            metrics = self.metric(pred_decoded, caption_decoded)

            # Accumulate loss and metrics
            total_loss += loss.item()
            total_metrics += metrics[self.kwargs.get('metric', 'bleu1')]

        # Compute average loss and metrics
        avg_loss = total_loss / num_batches
        avg_metrics = total_metrics / num_batches
        return avg_loss, avg_metrics
    
    def __eval_epoch(self, valid_loader: torch.utils.data.DataLoader):
        """Evaluates the model for one epoch using the provided data loader.

        Args:
            valid_loader (torch.utils.data.DataLoader): Data loader for validation.
        """
        self.model.eval()  # Ensure the model is in evaluation mode
        total_loss = 0
        total_metrics = 0
        num_batches = len(valid_loader)

        with torch.no_grad():  # No need to compute gradients for evaluation
            for img, caption in tqdm(valid_loader, desc='Validation', total=num_batches):
                img, caption = img.to(self.device), caption.to(self.device)

                # Forward pass
                pred = self.model(img)
                loss = self.criterion(pred, caption)

                # Compute the metrics
                pred_text = torch.argmax(pred, dim=1)  # Assuming the model generates token indices
                pred_text = pred_text.cpu().numpy()
                caption_text = caption.cpu().numpy()
                
                pred_decoded = self.tokenizer.decode(pred_text)
                caption_decoded = self.tokenizer.decode(caption_text)

                # Compute the metrics
                caption_decoded = np.array([[text] for text in caption_decoded], dtype=object)
                metrics = self.metric(pred_decoded, caption_decoded)

                # Accumulate loss and metrics
                total_loss += loss.item()
                total_metrics += metrics[self.kwargs.get('metric', 'bleu1')]

        # Compute average loss and metrics
        avg_loss = total_loss / num_batches
        avg_metrics = total_metrics / num_batches
        return avg_loss, avg_metrics