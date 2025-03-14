import os
import albumentations as A

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from week2.src.detectron.dataset import AlbumentationsMapper


def get_augmentations() -> A.Compose:
	"""Get the augmentations to apply.

	Returns:
		A.Compose: Compose of augmentations from albumentations.
	"""
	return A.Compose([
		A.MotionBlur(p=0.25, blur_limit=(3, 8)),
  		A.Illumination(p=0.4, intensity_range=(0.1, 0.2)),
        A.AtLeastOneBBoxRandomCrop(p=0.2, height=185, width=613),
        A.Rotate(p=0.3, limit=(-5, 5))
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
		# return build_detection_train_loader(cfg)
