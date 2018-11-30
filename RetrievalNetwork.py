import numpy as np, os, random
import scipy.io as sio
from PIL import Image
import cv2
os.environ['PYTHONHASHSEED'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

np.random.seed(1)
random.seed(1)

from keras.layers.core import Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

from keras.models import Model, Sequential, load_model

data_dir = ''
flip = False
views = []
samples_num = 0
val_samples_num = 0
num_epochs = 0
batch_size = 0
input_dim = ()


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network(train_last_layer=False):
    model = Sequential()

    model.add(Conv2D(128, (7, 7), padding='same', input_shape=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5, seed=1))

    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5, seed=1))

    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5, seed=1))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5, seed=1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.layers[-1].trainable = not train_last_layer
    return model


def create_model(pre_trained_model=None):
    K.set_image_dim_ordering('th')

    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)

    processed_a = create_base_network(pre_trained_model is None)(input_a)
    processed_b = create_base_network(pre_trained_model is None)(input_b)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    if pre_trained_model is not None:
        model.load_weights(pre_trained_model)
    return model


def data_generator(partial_path='Training'):
    def return_sample(sample_path):
        global input_dim
        if '.mat' in sample_path:   # Optical Flow
            return cv2.normalize(np.array(sio.loadmat(sample_path)['opt_flow']), None, 0, 255, cv2.NORM_MINMAX)
        else:   # Image
            return np.rollaxis(cv2.normalize(np.array(Image.open(sample_path)), None, 0, 255, cv2.NORM_MINMAX), 2, 0)
    imgs_dirs = os.listdir(data_dir+'/'+partial_path+'/Camera_1')
    while 1:
        x, y = [], []
        for j in range(batch_size):
            first_idx = random.randint(0, len(imgs_dirs)-1)
            img_1 = return_sample(data_dir+'/'+partial_path+'/Camera_'+str(views[0])+'/'+imgs_dirs[first_idx])
            if random.random() > 0.5:
                img_2 = return_sample(data_dir+'/'+partial_path+'/Camera_'+str(views[1])+'/'+imgs_dirs[first_idx])
                y.append(1)
            else:
                second_idx = random.randint(0, len(imgs_dirs)-1)
                img_2 = return_sample(data_dir+'/'+partial_path+'/Camera_'+str(views[1])+'/'+imgs_dirs[second_idx])
                y.append(0)

            if flip and random.random() > 0.5:
                cur_input = np.flip(np.array([img_1, img_2]), axis=-2).tolist()
            else:
                cur_input = [img_1, img_2]
            x.append(cur_input)
        yield ([np.array(x).swapaxes(0, 1)[0], np.array(x).swapaxes(0, 1)[1]], np.array(y))


def fit_model_generator(output_model_path, model, learning_rate):
    model.compile(loss=contrastive_loss, optimizer=RMSprop(lr=learning_rate))
    learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-12, verbose=1)
    model_checkpoint = ModelCheckpoint(output_model_path, save_best_only=True, save_weights_only=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto')
    csv_logger = CSVLogger(output_model_path.replace('.h5', '_log.csv'), append=True, separator=';')

    model.fit_generator(generator=data_generator(),
                        samples_per_epoch=int(samples_num / batch_size),
                        epochs=num_epochs, verbose=2,
                        callbacks=[early_stopping, model_checkpoint,
                                   learning_rate_scheduler, csv_logger],
                        validation_data=data_generator('Validation'),
                        validation_steps=val_samples_num, max_queue_size=30)
    return model


def start_training(data_path, output_model_path, pre_trained_model_path=None,
                   learning_rate=1e-4, vs=[1, 2], num_samples=0, size_batch=16,
                   val_num_samples=0, images_size=(3, 224, 224), epochs_num=100):
    global views, data_dir, samples_num, val_samples_num, flip, input_dim, num_epochs, batch_size
    print data_path
    views = vs
    input_dim = images_size
    num_epochs = epochs_num
    batch_size = size_batch
    data_dir = data_path

    if 'Real' in data_path:
        flip = True

    if num_samples > 0:
        samples_num = num_samples
    else:
        samples_num = len(os.listdir(data_path + 'Training/Camera_1'))

    if val_num_samples > 0:
        val_samples_num = val_num_samples
    else:
        val_samples_num = len(os.listdir(data_path + 'Validation/Camera_1'))

    model = create_model(pre_trained_model=pre_trained_model_path)
    fit_model_generator(output_model_path, model, learning_rate)
