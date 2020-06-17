# coding: utf-8

import pickle, os
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve,auc,recall_score,precision_score,f1_score,average_precision_score
import numpy as np
from collections import OrderedDict

params = {
    'n_layers':12,
    'lr':10**-0,
    'n_int_filters':7,
    'n_final_layer':54,
    'batch_size':2**4,
    'dropout_strength':0.288920,
}
params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))

h,w = 125,125

def add_block(x):
    for i in range(3):
        x = Conv2D(params['n_int_filters'],3,padding='same',kernel_initializer = 'he_normal')(x)    
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
for i in range(params['n_layers']-1):
    x = Concatenate()(layer_list)
    x = add_block(x)
    layer_list.append(x)

x = Dropout(params['dropout_strength'])(x)
x = Conv2D(params['n_final_layer'], (3, 3), dilation_rate=(2,2),padding='same')(x)
x = Conv2D(1,1,activation = 'sigmoid')(x)

model = Model(inputs = input,outputs = [x])
sgd = SGD(nesterov = True,decay=1E-6,lr=.01)
model.summary()
print('MODEL PARAMS:',params)
model.compile(loss=['binary_crossentropy'], optimizer=sgd,
             loss_weights = [1],metrics=['accuracy'])

initial_epoch = 0

id_str = '_'.join([str(i) for i in params.values()])

csv_logger = CSVLogger('binary_localizer_{}.log'.format(id_str))
tbCallBack = TensorBoard(log_dir='tensorboard/{}_{}'.format(id_str,str(initial_epoch)), 
    histogram_freq=0, write_graph=True, write_images=True)
save_loc = 'binary_localizer_{}.hdf5'.format(id_str)
checkpointer = ModelCheckpoint(filepath=save_loc, verbose=1, save_best_only=True)

data_loc = os.path.join('Data', 'binary_data.p')
x_train,x_val,y_train,y_val = pickle.load(open(data_loc,'rb'))

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

image_generator = image_datagen.flow(x_train, seed=seed, batch_size=params['batch_size'])
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=params['batch_size'])

x_val_generator = image_datagen.flow(x_val, seed=seed, batch_size=100)
y_val_generator = mask_datagen.flow(y_val, seed=seed, batch_size=100)

train_generator = zip(image_generator, mask_generator)
val_generator = zip(x_val_generator,y_val_generator)


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
    # y_pred = np.round(y_pred)
    y_true = np.round(y_true)
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)        
    return  auc(fpr_keras, tpr_keras)

def get_f1(precision,recall):
    return 2*(precision*recall)/(precision+recall)

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

metrics = Metrics(val_generator,model,1)
early_stopping = EarlyStopping(patience = 10)
model.fit_generator(train_generator,
                    validation_data = val_generator,
                    validation_steps=100,
                    steps_per_epoch=400,
                    epochs=3000,
                    callbacks = [metrics,tbCallBack,early_stopping,checkpointer,csv_logger])
