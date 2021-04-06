# Import packages
from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile
import logging
from io import BytesIO

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

# from config import get_args
from config import config
# from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
# from lib.datasets.dataset import LmdbDataset, AlignCollate
# from lib.loss import SequenceCrossEntropyLoss
# from lib.trainers import Trainer
# from lib.evaluators import Evaluator
# from lib.utils.logging import Logger, TFLogger
# from lib.utils.serialization import load_checkpoint, save_checkpoint
# from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

# Set up logger
logger = logging.getLogger(__name__)
logger.info('Import packages done.')

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using {device}')

# Define some required class and function
class DataInfo(object):
	"""
	Save the info about the dataset.
	This a code snippet from dataset.py
	"""
	def __init__(self, voc_type):
		super(DataInfo, self).__init__()
		self.voc_type = voc_type

		assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
		self.EOS = 'EOS'
		self.PADDING = 'PADDING'
		self.UNKNOWN = 'UNKNOWN'
		self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
		self.char2id = dict(zip(self.voc, range(len(self.voc))))
		self.id2char = dict(zip(range(len(self.voc)), self.voc))

		self.rec_num_classes = len(self.voc)

def _npy_load(data):
	stream = BytesIO(data)
	return np.load(stream)

def _npy_dump(data):
	buffer = BytesIO()
	np.save(buffer, data)
	return buffer.getvalue()

def image_process(image, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
	img = Image.fromarray(image)

	if keep_ratio:
		w, h = img.size
		ratio = w / float(h)
		imgW = int(np.floor(ratio * imgH))
		imgW = max(imgH * min_ratio, imgW)

	img = img.resize((imgW, imgH), Image.BILINEAR)
	img = transforms.ToTensor()(img)
	img.sub_(0.5).div_(0.5)

	return img

# Declare some variables

dataset_info = DataInfo(config['voc_type'])

# Load model
def model_fn(model_dir):
	logger.info('Inside model_fn function.')
	logger.info('Loading model...')

	model = ModelBuilder(arch=config['arch'], rec_num_classes=dataset_info.rec_num_classes,
                       sDim=config['decoder_sdim'], attDim=config['attDim'], max_len_labels=config['max_len'],
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=config['STN_ON'])

	model_path = f'{model_dir}/model.pth'
	model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
	model.eval()
	logger.info('Loading model done.')

	return model

# Processing input
def input_fn(request_body, content_type='application/x-npy'):
	if content_type == 'application/x-npy':
		logger.info(f'Content type is {content_type}.')
		logger.info('Convert request_body to numpy array...')
		# Load request body and convert to numpy array
		image = _npy_load(request_body)
		# Process data
		image = image_process(image)
		image = image.to(device)
		input_dict = {}
		input_dict['images'] = image.unsqueeze(0)
		# TODO: testing should be more clean.
		# to be compatible with the lmdb-based testing, need to construct some meaningless variables.
		rec_targets = torch.IntTensor(1, config['max_len']).fill_(1)
		rec_targets[:,config['max_len']-1] = dataset_info.char2id[dataset_info.EOS]
		input_dict['rec_targets'] = rec_targets
		input_dict['rec_lengths'] = [config['max_len']]

		return input_dict

	raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

# Inference
def predict_fn(input_obj, model):
	"""
	Get prediction
	"""
	logger.info('In predict_fn function.')
	model.to(device)

	with torch.no_grad():
		output_dict = model(input_obj)
		pred_rec = output_dict['output']['pred_rec']
		pred_str, _ = get_str_list(pred_rec, input_obj['rec_targets'], dataset=dataset_info)
		# print('Recognition result: {0}'.format(pred_str[0]))

		logger.info(f'Prediction: {pred_str}')

	return pred_str[0]