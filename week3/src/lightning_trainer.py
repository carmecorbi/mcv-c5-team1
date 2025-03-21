import torch
import numpy as np
import pytorch_lightning as pl

from lightning.pytorch.loggers import CSVLogger
from src.metrics.metrics import Metric


class LightningTrainer(pl.LightningModule):
    def __init__(
        self, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        tokenizer,
        optimizer_class=torch.optim.Adam,
        learning_rate=1e-3,
        **kwargs
    ):
        """
        PyTorch Lightning implementation of the trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            tokenizer: Tokenizer for text decoding
            optimizer_class: Optimizer class to use
            learning_rate: Learning rate for optimizer
            metric_name: Metric to track during training
            **kwargs: Additional parameters
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.metric = Metric()
        
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model', 'criterion', 'tokenizer'])
        
    def forward(self, x):
        """Forward pass of the model"""
        return self.model(x)
    
    def _compute_metrics(self, pred, caption):
        """Helper method to compute metrics"""
        pred_text = torch.argmax(pred, dim=1)
        pred_text = pred_text.cpu().numpy()
        caption_text = caption.cpu().numpy()
        
        # Decode token indices to text
        pred_decoded = self.tokenizer.decode(pred_text)
        caption_decoded = self.tokenizer.decode(caption_text)
        
        # Format for metric calculation
        caption_decoded = np.array([[text] for text in caption_decoded], dtype=object)
        
        print(pred_decoded)
        try:
            metrics = self.metric(pred_decoded, caption_decoded)
        except Exception as e:
            print(pred_decoded)
            print(caption_decoded)
            raise ValueError(f"Error in metric calculation: {e}")
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        img, caption = batch
        
        # Forward pass
        pred = self(img)
        
        # Compute loss
        loss = self.criterion(pred, caption)
        
        # Compute metrics
        metrics = self._compute_metrics(pred, caption)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'train_{metric_name}', metric_value, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        img, caption = batch
        
        # Forward pass
        pred = self(img)
        
        # Compute loss
        loss = self.criterion(pred, caption)
        
        # Compute metrics
        metrics = self._compute_metrics(pred, caption)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'train_{metric_name}', metric_value, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Single test step"""
        img, caption = batch
        
        # Forward pass
        pred = self(img)
        
        # Compute loss
        loss = self.criterion(pred, caption)
        
        # Compute metrics
        metrics = self._compute_metrics(pred, caption)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'train_{metric_name}', metric_value, prog_bar=True, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer"""
        return self.optimizer_class(self.parameters(), lr=self.learning_rate)


def train_with_lightning(
    model, 
    criterion, 
    tokenizer, 
    train_loader, 
    val_loader, 
    test_loader, 
    max_epochs=10,
    learning_rate=1e-3,
    save_dir='/ghome/c5mcv01/mcv-c5-team1/week3/results',
    exp_name='baseline',
    **kwargs
):
    """Helper function to train with Lightning"""
    # Set up logger
    logger = CSVLogger(save_dir, name=exp_name)
    
    # Create Lightning module
    lightning_model = LightningTrainer(
        model=model,
        criterion=criterion,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        **kwargs
    )
    
    # Create Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Automatically select GPU if available
        devices='auto',
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        ]
    )
    
    # Train the model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test the model
    trainer.test(dataloaders=test_loader)
    return lightning_model, trainer
