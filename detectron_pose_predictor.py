import os
import numpy as np
import cv2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def get_img_names(imgs_dir):
	img_names = []
	for dirpath, dirnames, filenames in os.walk(imgs_dir):
		for filename in [f for f in filenames if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]:
			img_names.append(filename)
	img_names.sort()

	return img_names


def init_pose_predictor(config_path, weights_path, cuda=True):
	cfg = get_cfg()
	cfg.merge_from_file(config_path)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = weights_path
	if cuda == False:
		cfg.MODEL.DEVICE='cpu'
	predictor = DefaultPredictor(cfg)

	return predictor


def encode_for_videpose3d(boxes,keypoints,resolution, dataset_name):
	# Generate metadata:
	metadata = {}
	metadata['layout_name'] = 'coco'
	metadata['num_joints'] = 17
	metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
	metadata['video_metadata'] = {dataset_name: resolution}

	prepared_boxes = []
	prepared_keypoints = []
	for i in range(len(boxes)):
		if len(boxes[i]) == 0 or len(keypoints[i]) == 0:
			# No bbox/keypoints detected for this frame -> will be interpolated
			prepared_boxes.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
			prepared_keypoints.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
			continue

		prepared_boxes.append(boxes[i])
		prepared_keypoints.append(keypoints[i][:,:2])
		
	boxes = np.array(prepared_boxes, dtype=np.float32)
	keypoints = np.array(prepared_keypoints, dtype=np.float32)
	keypoints = keypoints[:, :, :2] # Extract (x, y)
	
	# Fix missing bboxes/keypoints by linear interpolation
	mask = ~np.isnan(boxes[:, 0])
	indices = np.arange(len(boxes))
	for i in range(4):
		boxes[:, i] = np.interp(indices, indices[mask], boxes[mask, i])
	for i in range(17):
		for j in range(2):
			keypoints[:, i, j] = np.interp(indices, indices[mask], keypoints[mask, i, j])
	
	print('{} total frames processed'.format(len(boxes)))
	print('{} frames were interpolated'.format(np.sum(~mask)))
	print('----------')
	
	return [{
		'start_frame': 0, # Inclusive
		'end_frame': len(keypoints), # Exclusive
		'bounding_boxes': boxes,
		'keypoints': keypoints,
	}], metadata


def predict_pose(pose_predictor, imgs_dir, output_path, dataset_name='detectron2'):
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

	# Set metadata:
	img = cv2.imread(os.path.join(imgs_dir, img_names[0]))
	resolution = {
		'w': img.shape[1],
		'h': img.shape[0],
	}

	# Encode data in VidePose3d format and save it as a compressed numpy (.npz):
	data, metadata = encode_for_videpose3d(boxes, keypoints, resolution, dataset_name)
	output = {}
	output[dataset_name] = {}
	output[dataset_name]['custom'] = [data[0]['keypoints'].astype('float32')]
	np.savez_compressed(output_path, positions_2d=output, metadata=metadata)

	print ('All done!')


if __name__ == '__main__':
	# Init pose predictor:
	model_config_path = './keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
	model_weights_path = './model_final_5ad38f.pkl'	
	pose_predictor = init_pose_predictor(model_config_path, model_weights_path, cuda=False)

	# Predict poses and save the result:
	imgs_dir = './images'
	output_path = './pose2d'
	predict_pose(pose_predictor, imgs_dir, output_path)





