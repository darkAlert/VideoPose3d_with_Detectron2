import os
import numpy as np
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def get_img_names(imgs_dir):
    img_names = []
    for dirpath, dirnames, filenames in os.walk(imgs_dir):
        for filename in [f for f in filenames if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]:
            img_names.append(filename)
    img_names.sort()

    return img_names


def init_pose_predictor(config_path, weights_path):
	cfg = get_cfg()
	cfg.merge_from_file(config_path)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = weights_path
	predictor = DefaultPredictor(cfg)

	return predictor


def predict_pose(pose_predictor, imgs_dir, output_path):
	'''
		pose_predictor: The detectron's pose predictor
		imgs_dir:       The path to the directory containing the images
		output_path:    The path where the result will be saved in .npz format
	'''

	img_names = get_img_names(imgs_dir)
	n = len(img_names)
	boxes = []
	keypoints = []

	# Predict poses:
	for i, name in enumerate(img_names):
		img_path = os.path.join(imgs_dir, name)
		img = cv2.imread(img_path)
		pose_output = pose_predictor(img)

		if len(pose_output["instances"].pred_boxes.tensor) > 0:
			cls_boxes = pose_output["instances"].pred_boxes.tensor[0].cpu().numpy()
			cls_keyps = pose_output["instances"].pred_keypoints[0].cpu().numpy()
		else:
			cls_boxes = np.full((4,), np.nan, dtype=np.float32)
			cls_keyps = np.full((17,3), np.nan, dtype=np.float32)   # nan for images that do not contain human

		boxes.append(cls_boxes)
		keypoints.append(cls_keyps)

		print('{}/{}      '.format(i+1, n), end='\r')
	print('\n')

	# Set metadata:
	img = cv2.imread(os.path.join(imgs_dir, img_names[0]))
	metadata = {
		'w': img.shape[1],
		'h': img.shape[0],
	}

	# Save result:
	np.savez_compressed(output_path, boxes=boxes, keypoints=keypoints, metadata=metadata)
	print ('All done!')


if __name__ == '__main__':
	# Init pose predictor:
	model_config_path = '/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
	model_weights_path = '/model_final_5ad38f.pkl'	
	pose_predictor = init_pose_predictor(model_config_path, model_weights_path)

	# Predict poses and save the result:
	imgs_dir = '/images'
	output_path = '/pose2d'
	predict_pose(pose_predictor, imgs_dir, output_path)
