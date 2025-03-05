from ultralytics import YOLO
import time

model = YOLO('/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt')

start_time = time.time()
# Train the model
results = model.train(
        data='/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cuda',
        patience=20,  # Early stopping patience,
        project='finetune_unfrozen_epoch70',
        freeze=0,
        classes=[0,2],
        mixup=0.39,
        optimizer='SGD',
        dropout=0.32,
        weight_decay=9.37e-5,
        degrees=0.32,
        scale=0.88
    )

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
