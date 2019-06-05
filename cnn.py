import os
import cv2
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import keras
import sklearn
import glob
from  sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import imgaug as ia
from keras.applications.nasnet import preprocess_input,NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#print(os.listdir('E:/project/kaggle/cancer'))

#读取文件
df_train = pd.read_csv('train_labels.csv')
#id 和 label进行映射
id_label_map = {k:v for k,v in zip(df_train.id.values, df_train.label.values)}
#print(df_train.head(5))
#print(df_train.info()) 220025 id/label

#加载图片目录
labeled_files = glob.glob('train/*.tif')
test_files = glob.glob('test/*.tif')
#print(len(labeled_files)) 220025
#print(len(test_files)) 57458

#获取图片id
def get_id_from_file_path(file_path):
    return file_path.split('.')[0].split('\\')[-1]

#划分训练和验证集
train,val = train_test_split(labeled_files,test_size=0.1,random_state=42)


def get_seq():
    sometimes = lambda aug:iaa.Sometimes(0.5,aug)
    seq = iaa.Sequential(
        [iaa.Fliplr(0.5),
         iaa.Flipud(0.2),
         sometimes(iaa.Affine(
             scale = {'x':(0.9,1.1),'y':(0.9,1.1)},
            translate_percent = {'x':(-0.1,0.1),'y':(-0.1,0.1)},
             rotate = (-10,10),
             shear = (-5,5),
             order = [0,1],
             cval = (0,255),
             mode = ia.ALL,
         )),
         iaa.SomeOf((0,5),
                    [sometimes(iaa.Superpixels(p_replace=(0,1.0),n_segments=(20,200))),
                    iaa.OneOf([iaa.GaussianBlur((0,1.0)),
                               iaa.AverageBlur(k=(3,5)),
                               iaa.MedianBlur(k=(3,5)),
                    ]),
                     iaa.Sharpen(alpha=(0,1.0),lightness=(0.9,1.1)),
                     iaa.Emboss(alpha=(0,1.0),strength=(0,2.0)),
                     iaa.SimplexNoiseAlpha(iaa.OneOf([iaa.EdgeDetect(alpha=(0.5,1.0)),
                                            iaa.DirectedEdgeDetect(alpha=(0.5,1.0),
                                                                   direction=(0.0,1.0))
                     ])),
                     iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.01*255),per_channel=0.5),
                     iaa.OneOf([
                                iaa.Dropout((0.01,0.05),per_channel=0.5,),
                                iaa.CoarseDropout((0.01,0.03),size_percent=(0.01,0.02),per_channel=0.2)
                                ]),
                     iaa.Invert(0.01,per_channel=True),
                     iaa.Add((-2,2),per_channel=0.5),
                     iaa.AddToHueAndSaturation((-1,1)),
                     iaa.OneOf([
                                iaa.Multiply((0.9,1.1),per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(exponent=(-1,0),first=iaa.Multiply((0.9,1.1),per_channel=True),
                                                        second=iaa.ContrastNormalization((0.9,1.1)))
                     ]),
                     sometimes(iaa.ElasticTransformation(alpha=(0.5,3.5),sigma=0.25)),
                     sometimes(iaa.PiecewiseAffine(scale=(0.01,0.05))),
                     sometimes(iaa.PerspectiveTransform(scale=(0.01,0.1)))
                                ],
                        random_order=True)],
        random_order=True)
    return seq

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def data_gen(list_files,id_label_map,batch_size,augment=False):
    seq = get_seq()
    while True:
        shuffle(list_files)
        for batch in chunker(list_files,batch_size):
            X = [cv2.imread(x) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]
            if augment:
                X = seq.augment_images(X)
            X = [preprocess_input(x) for x in X]

            yield np.array(X),np.array(Y)

def get_model_classify_nasnet():
    inputs = Input((96,96,3))
    base_model = NASNetMobile(include_top = False,input_shape=(96,96,3),weights=None)
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1,out2,out3])
    out = Dropout(0.5)(out)
    out = Dense(units=1,activation='sigmoid',name='3_')(out)
    model = Model(inputs,out)
    model.compile(optimizer=Adam(0.0001),loss=binary_crossentropy,metrics=['acc'])
    model.summary()

    return model

model = get_model_classify_nasnet()

batch_size = 32
h2_path = 'model.h5'
checkpoint = ModelCheckpoint(filepath=h2_path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
history = model.fit_generator(data_gen(train,id_label_map,batch_size,augment=True),
                              validation_data=data_gen(val,id_label_map,batch_size),
                              epochs=2,
                              verbose=1,
                              callbacks=[checkpoint],
                              steps_per_epoch=len(train)//batch_size,
                              validation_steps=len(val)//batch_size)

#model.load_weights(h2_path)
'''










