import os
import albumentations as A

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from week1.src.detectron.dataset import AlbumentationsMapper


def get_augmentations():
    return A.Compose([
        # A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(p=0.25),
        A.RandomBrightnessContrast(p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=100, min_visibility=0.1, label_fields=['category_ids']))

class CustomTrainer(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		if output_folder is None:
			output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
		return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
	
	@classmethod
	def build_train_loader(cls, cfg):
		mapper = AlbumentationsMapper(cfg, is_train=True, augmentations=get_augmentations())
		return build_detection_train_loader(cfg, mapper=mapper)
