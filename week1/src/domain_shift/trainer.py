import os
import albumentations as A

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from dataset import AlbumentationsMapper


def get_augmentations() -> A.Compose:
	"""Get the augmentations to apply.

	Returns:
		A.Compose: Compose of augmentations from albumentations.
	"""
	return A.Compose([
        A.OneOf([
			A.RandomCrop(width=500, height=500),
			A.RandomCrop(width=400, height=400),
			A.RandomCrop(width=450, height=450),
		], p=0.6),
        A.HorizontalFlip(p=0.6),
        A.ShiftScaleRotate(p=0.25),
        A.RandomBrightnessContrast(p=0.3),
		A.HueSaturationValue(p=0.3),
		A.GaussianBlur(blur_limit=3, p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.1, label_fields=['category_ids']))

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
