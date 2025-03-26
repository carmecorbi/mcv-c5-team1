import torch
import numpy as np
import pytorch_lightning as pl
import pandas as pd

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
        scheduler_type=None,
        scheduler_params=None,
        use_teacher_forcing=False,
        teacher_forcing_ratio=0.5,
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
        self.use_teacher_forcing = use_teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        self.kwargs = kwargs
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model', 'criterion', 'tokenizer'])
        
    def forward(self, x, caption=None, teacher_forcing_ratio=0.5):
        """Forward pass of the model"""
        if self.use_teacher_forcing:
            return self.model(x, caption, teacher_forcing_ratio)
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
        if self.use_teacher_forcing:
            pred = self(img, caption, self.teacher_forcing_ratio)
        else:
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
            self.log(f'val_{metric_name}', metric_value, prog_bar=True, on_epoch=True)
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
            self.log(f'test_{metric_name}', metric_value, prog_bar=True, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        
        if self.scheduler_type is None:
            return optimizer
        
        scheduler_config = {
            "optimizer": optimizer,
        }
        
        if self.scheduler_type == 'step':
            step_size = self.scheduler_params.get("step_size", 10)
            gamma = self.scheduler_params.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=step_size,
                gamma=gamma
            )
            scheduler_config["lr_scheduler"] = scheduler
        elif self.scheduler_type == 'cosine':
            t_max = self.scheduler_params.get("T_max", self.trainer.max_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max
            )
            scheduler_config["lr_scheduler"] = scheduler
        elif self.scheduler_type == 'linear':
            # Linear scheduler using OneCycleLR with linear anneal
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                anneal_strategy='linear'
            )
            scheduler_config["lr_scheduler"] = scheduler
            scheduler_config["interval"] = "step"
        return scheduler_config


def train_with_lightning(
    model, 
    criterion, 
    tokenizer, 
    train_loader, 
    val_loader, 
    test_loader, 
    max_epochs=10,
    optimizer_name='adam',
    learning_rate=1e-3,
    use_teacher_forcing=False,
    teacher_forcing_ratio=0.5,
    gradient_clip_val=None,
    scheduler_type=None,
    scheduler_params=None,
    save_dir='/ghome/c5mcv01/mcv-c5-team1/week3/results',
    exp_name='baseline',
    early_stopping_criteria='train_loss',
    **kwargs
):
    """Helper function to train with Lightning"""
    # Set up logger
    logger = CSVLogger(save_dir, name=exp_name)
    
    if optimizer_name == 'adam':
        optimizer_class = torch.optim.Adam
    elif optimizer_name == 'sgd':
        optimizer_class = torch.optim.SGD
    elif optimizer_name == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    
    # Create Lightning module
    lightning_model = LightningTrainer(
        model=model,
        criterion=criterion,
        tokenizer=tokenizer,
        optimizer_class=optimizer_class,
        learning_rate=learning_rate,
        use_teacher_forcing=use_teacher_forcing,
        teacher_forcing_ratio=teacher_forcing_ratio,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params,
        **kwargs
    )
    
    # Create Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Automatically select GPU if available
        devices='auto',
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True),
            pl.callbacks.EarlyStopping(monitor=early_stopping_criteria, patience=5, mode='min')
        ]
    )
    
    # Train the model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Get the val loss from loggers metrics
    csv_path = f"{save_dir}/{exp_name}/version_0/metrics.csv"
    metrics_df = pd.read_csv(csv_path)
    print(metrics_df.columns)
    val_loss = metrics_df['val_loss'].dropna()  # Drop NaN values (if any)
    best_val_loss = val_loss.min()
    
    # Test the model
    trainer.test(dataloaders=test_loader)
    return lightning_model, trainer, best_val_loss
