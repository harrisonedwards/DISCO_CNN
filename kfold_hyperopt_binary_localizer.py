# coding: utf-8

import pickle,os
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve,auc,recall_score,precision_score,f1_score,average_precision_score
import numpy as np
from collections import OrderedDict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
import time

def get_data(split_n,params):
	data_loc = 'binary_data.p'
	x_train,x_val,y_train,y_val = pickle.load(open(data_loc,'rb'))
	x_train = np.concatenate((x_train,x_val), 0)
	y_train = np.concatenate((y_train,y_val), 0)

	num_train = int(np.round(x_train.shape[0]*.66))
	num_val = int(np.round(x_train.shape[0]*.34))
	
	split_dict = {1: (x_train[:num_train],x_train[-num_val:],y_train[:num_train],y_train[-num_val:]),
	# middle split..
	2: (x_train[int(num_val/2):-int(num_val/2)],np.concatenate((x_train[:int(num_val/2)],x_train[int(-num_val/2):])),
	    y_train[int(num_val/2):-int(num_val/2)],np.concatenate((y_train[:int(num_val/2)],y_train[int(-num_val/2):]))),
	3: (x_train[-num_train:],x_train[:num_val],y_train[-num_train:],y_train[:num_val])
	}

	x_train,x_val,y_train,y_val = split_dict[split_n]

	seed = 1
	image_gen_args = dict(rotation_range = 90.,
						 width_shift_range = 0.05,
						 height_shift_range = 0.05,
						 vertical_flip = True,
						 horizontal_flip = True) 

	mask_gen_args = dict(rotation_range = 90.,
						 width_shift_range = 0.05,
						 height_shift_range = 0.05,
						 vertical_flip = True,
						 horizontal_flip = True)  

	image_datagen = ImageDataGenerator(**image_gen_args) 
	mask_datagen = ImageDataGenerator(**mask_gen_args)

	image_generator = image_datagen.flow(x_train, seed=seed, batch_size=int(params['batch_size']))
	mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=int(params['batch_size']))

	x_val_generator = image_datagen.flow(x_val, seed=seed, batch_size=100)
	y_val_generator = mask_datagen.flow(y_val, seed=seed, batch_size=100)

	train_generator = zip(image_generator, mask_generator)
	val_generator = zip(x_val_generator,y_val_generator)

	return train_generator,val_generator

def precision(y_true, y_pred):
		'''Calculates the precision, a metric for multi-label classification of
		how many selected items are relevant.
		'''
		true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
		predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + 1E-7)
		return precision

def recall(y_true, y_pred):
	'''Calculates the recall, a metric for multi-label classification of
	how many relevant items are selected.
	'''
	true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
	possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + 1E-7)
	return recall


def mean_iou( y_true, y_pred, num_classes):
	# Compute the confusion matrix to get the number of true positives,
	# false positives, and false negatives
	# Convert predictions and target from categorical to integer format
	target = np.argmax(y_true, axis=-1).ravel()
	predicted = np.argmax(y_pred, axis=-1).ravel()

	# Trick from torchnet for bincounting 2 arrays together
	# https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
	x = predicted + num_classes * target
	bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes**2)
	assert bincount_2d.size == num_classes**2
	conf = bincount_2d.reshape((num_classes, num_classes))

	# Compute the IoU and mean IoU from the confusion matrix
	true_positive = np.diag(conf)
	false_positive = np.sum(conf, 0) - true_positive
	false_negative = np.sum(conf, 1) - true_positive

	# Just in case we get a division by 0, ignore/hide the error and set the value to 0
	with np.errstate(divide='ignore', invalid='ignore'):
		iou = true_positive / (true_positive + false_positive + false_negative)
	iou[np.isnan(iou)] = 0

	return np.mean(iou).astype(np.float32)

def get_auc(y_true,y_pred):
	y_true = np.round(y_true)
	y_pred = y_pred.ravel()
	y_true = y_true.ravel()
	fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)        
	return  auc(fpr_keras, tpr_keras)

def get_f1(precision,recall):
	return 2*(precision*recall)/(precision+recall)+1E-7

def get_average_precision_score(y_true,y_pred):
	y_true = y_true.ravel()
	y_pred = y_pred.ravel()
	y_true = np.round(np.clip(y_true, 0, 1))
	return average_precision_score(y_true,y_pred)

class Metrics(keras.callbacks.Callback):
	def __init__(self,validation_data,model,num_classes):
		self.validation_data = validation_data
		self.model = model
		self.num_classes = num_classes
		
	def on_epoch_end(self, batch, logs={}):
		x_val,y_true = next(self.validation_data)
		y_pred = self.model.predict(x_val)
		self.recall_score = recall(y_true, y_pred)
		logs['recall_score'] = self.recall_score        
		self.precision_score = precision(y_true, y_pred)
		logs['precision_score'] = self.precision_score 
		self.auc_score = get_auc(y_true, y_pred)
		logs['auc_score'] = self.auc_score 
		self.f1_score = get_f1(self.precision_score,self.recall_score)
		logs['f1_score'] = self.f1_score 
		self.average_precision_score = get_average_precision_score(y_true, y_pred)
		logs['average_precision_score'] = self.average_precision_score 
		# self.mean_iou_score = mean_iou(y_true, y_pred, self.num_classes)
		# logs['mean_iou'] = self.mean_iou_score
		print(logs)
		return


def create_model(params):	

	h,w = 125,125

	def add_block(x):
		for i in range(3):
			x = Conv2D(int(params['n_filters']),3,padding='same',kernel_initializer = 'he_normal')(x)    
			x = LeakyReLU()(x)
			x = BatchNormalization()(x)
		return x 

	input = Input(shape=(h,w,1))
	x1 = Conv2D(16,3,padding='same',kernel_initializer = 'he_normal')(input)    
	x1 = LeakyReLU()(x1)
	x1 = BatchNormalization()(x1)
	x2 = Conv2D(16,3,padding='same',kernel_initializer = 'he_normal')(x1)    
	x2 = LeakyReLU()(x2)
	x2 = BatchNormalization()(x2)
	layer_list = [x1,x2]
	for i in range(int(params['n_layers'])):
		x = Concatenate()(layer_list)
		x = add_block(x)
		layer_list.append(x)

	x = Dropout(params['droput_strength'])(x)
	x = Conv2D(int(params['n_final_layer']), (3, 3), dilation_rate=(2,2),padding='same')(x)
	x = Conv2D(1,1,activation = 'sigmoid')(x)

	model = Model(inputs = input,outputs = [x])
	
	sgd = SGD(nesterov = True,decay=1E-6,lr=params['learning_rate'])
	# model.summary()
	
	model.compile(loss=['binary_crossentropy'], optimizer=sgd,
				 loss_weights = [1])
	
	return model 

def objective(params):
		genetic_loss = 0
		average_precision_score = 0
		auc_score = 0
		precision_score = 0
		recall_score = 0
		f1_score = 0		

		for i in range(1,4):			
			model = None
			model = create_model(params)
			train_generator,val_generator = get_data(i,params)
			metrics = Metrics(val_generator,model,1)
			model.fit_generator(train_generator,
						steps_per_epoch=400,
						epochs=5,
						callbacks = [metrics])
			genetic_loss += -(metrics.average_precision_score + metrics.auc_score)
			average_precision_score += metrics.average_precision_score
			auc_score += metrics.auc_score
			precision_score += metrics.precision_score
			recall_score += metrics.recall_score
			f1_score += metrics.f1_score

		genetic_loss /= 3
		average_precision_score /= 3
		auc_score /= 3
		precision_score /= 3 
		recall_score /= 3
		f1_score /= 3	

		# metrics calculated by the end of the 5th epoch will be used for optimization
		print('METRICS:',metrics.average_precision_score,metrics.auc_score)
		print('GENETIC LOSS:',genetic_loss)

		return {'loss': genetic_loss,
				'status': STATUS_OK,
				'average_precision_score': metrics.average_precision_score,
				'auc_score': metrics.auc_score,
				'precision_score':metrics.precision_score,
				'recall_score':metrics.recall_score,
				'f1_score':metrics.f1_score
				}

params = {
	'n_filters': hp.quniform('n_filters',1,20,1),
	'n_layers': hp.quniform('n_layers',1,16,1),
	'droput_strength': hp.uniform('droput_strength',0,1),
	'n_final_layer': hp.quniform('n_final_layer',1,256,1),
	'learning_rate': 10**hp.quniform('learning_rate',-3,2,1),
	'batch_size': 2**hp.quniform('batch_size',0,4,1)
}


if __name__ == '__main__':   	
	trial_loc = 'genetic_trials_cv.p'
	additional_evals = 3
	if os.path.isfile(trial_loc):
		trials = pickle.load(open(trial_loc,'rb'))
		max_evals = len(trials.trials) + additional_evals
	else:
		trials = Trials()
		max_evals = additional_evals
	while True:

		best = fmin(objective, space=params, algo=tpe.suggest, max_evals=max_evals, trials=trials)        
		print('DUMPING TRIALS...')    	
		# print(trials)
		# print(trials.trials)
		for trial in trials.trials:
			if 'result' in trial.keys():
				trial['result'].pop('model', None)
		pickle.dump(trials,open(trial_loc,'wb'))
		max_evals += additional_evals
	# X_train, Y_train, X_test, Y_test = data()
	# print("Evalutation of best performing model:")
	# print(best_model.evaluate(X_test, Y_test))
	# print("Best performing model chosen hyper-parameters:")
	# print(best_run)    
