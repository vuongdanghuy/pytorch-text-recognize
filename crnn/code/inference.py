import torch
from torch.autograd import Variable
import os
import json
import logging
from io import BytesIO
import numpy as np
from PIL import Image

import crnn
import utils
import dataset

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = dataset.resizeNormalize((100, 32))

converter = utils.strLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')

def model_fn(model_dir):
	logger.info('In model_fn')
	logger.info(f'List file: {os.listdir(model_dir)}')

	logger.info(f'Using device {device}')

	model = crnn.CRNN(32,1,37,256)

	logger.info('Loading model...')

	model_path = f'{model_dir}/model.pth'
	model.load_state_dict(torch.load(model_path, map_location=device))

	logger.info('Loading model done.')

	return model

def _npy_load(data):
	stream = BytesIO(data)
	return np.load(stream)

def _npy_dump(data):
	buffer = BytesIO()
	np.save(buffer, data)
	return buffer.getvalue()


def input_fn(request_body, content_type='application/json'):
	logger.info('In input_fn')
	logger.info('Deserializing the input data.')
	
	if content_type == 'application/x-npy':
		logger.info(f'Content type is {content_type}.')
		logger.info('Convert request_body to numpy array...')
		
		image = _npy_load(request_body)
		image = Image.fromarray(image)
		image = transformer(image)
		image = image.to(device)
		image = image.view(1, *image.size())

		return Variable(image)

	raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_obj, model):
	logger.info('In predict_fn')

	with torch.no_grad():
		preds = model(input_obj)
		_, preds = preds.max(2)
		preds = preds.transpose(1, 0).contiguous().view(-1)
		preds_size = Variable(torch.IntTensor([preds.size(0)]))

		logger.info(f'Prediction value: {preds}')

		raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
		sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
		
	return raw_pred, sim_pred