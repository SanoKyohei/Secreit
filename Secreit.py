from keras import models
from keras import layers
import pickle
from keras.applications import VGG16
from keras import optimizers
from keras.preprocessing import image
import numpy as np
from keras.layers import Dot
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import cv2
import random

def softmax(x):
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x), axis=1, keepdims=True)
    ## nan is caused by inf
    y[np.isnan(y)] = 1
    return y

def vgg_model(weight_path, input_shape=(240,320,3), input_tensor=None, pooling=None,classes=3):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor 
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x= Flatten(name='flatten')(x)
    x= Dense(500, activation='relu')(x)
    x= Dense(3, activation='linear')(x)
   #bilinear CNN
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model=Model(img_input, x, name='seq')
    # load weights
    model.load_weights(weight_path)
    return model


def predict(img, model):
    img=np.array(img)/255
    img1=img[0:240,0:320,:]    
    img2=img[0:240,320:640,:]
    img3=img[240:480,0:320,:]
    img4=img[240:480,320:640,:]        
    pre=softmax(model.predict_on_batch(np.array([img1, img2, img3, img4])))
    return np.mean(pre, axis=0)

def grad(img, lay, num, stage, seq3):
    x=np.array(img)/255
    P_output = seq3.output[:, stage]
    last_conv_layer = seq3.get_layer(lay)
    grads = K.gradients(P_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([seq3.input],[pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for ii in range(num):
        conv_layer_output_value[:, :, ii] *= pooled_grads_value[ii]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def Cam(img,stage, seq3):
    if stage=="D":
        stage=0
    elif stage=="E":
        stage=1
    elif stage=="P":
        stage=2
    
    x=img
    x1=x[0:240,0:320,:]
    x2=x[0:240,320:640,:]
    x3=x[240:480,0:320,:]
    x4=x[240:480,320:640,:]
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x3 = np.expand_dims(x3, axis=0)
    x4 = np.expand_dims(x4, axis=0)
    heat1=grad(x1,'block5_conv3' , 512, stage, seq3)
    heat2=grad(x2,'block5_conv3' , 512, stage, seq3)
    heat3=grad(x3,'block5_conv3' , 512, stage, seq3)
    heat4=grad(x4,'block5_conv3' , 512, stage, seq3)
    heatmap=np.zeros([30,40])
    heatmap[0:15,0:20]=heat1
    heatmap[0:15,20:40]=heat2
    heatmap[15:30,0:20]=heat3
    heatmap[15:30,20:40]=heat4
    img1=img[:,:,[2,1,0]]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img

class TrainGenerator(object):
    def __init__(self, trainD, trainE, trainP):
        self.reset()
        self.trainD=trainD
        self.trainE=trainE
        self.trainP=trainP

    def reset(self):
        self.images = []
        self.labels = []
        self.l=0
        
    def gene(self):
        train_datagen = ImageDataGenerator(
            featurewise_center=True, 
            zca_whitening=True, 
            vertical_flip=True,
            horizontal_flip=True,
            channel_shift_range=20,
            brightness_range=[0.5, 1.0],         
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,

            fill_mode="constant",
            cval=self.l )
        return train_datagen


    def flow_from_directory(self, batch_size=36):
        # LabelEncode(classをint型に変換)するためのdict
            # ディレクトリから画像のパスを取り出す
        ans_file=[self.trainD, self.trainE, self.trainP]
        
        while(1>0):
            #tはD,E,PÔのどのファイルにするか。kはファイル番号
            t=random.randint(0,2)
            k=random.randint(0,len(ans_file[t])-1)
            
            img=image.load_img(ans_file[t][k], target_size=(240, 320))
            img=image.img_to_array(img)
            self.l=np.mean(img)

            train_datagen =self.gene()

            img= img.reshape((1,) + img.shape)
            img=train_datagen.flow(img, batch_size=1)[0]
            
            img=img.reshape(240,320,3)
            
            img=img/255

            self.images.append(img)
            if t==0:
                self.labels.append([1,0,0])

            if t==1:
                self.labels.append([0,1,0])

            if t==2:
                self.labels.append([0,0,1])

            # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納
            # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする
            if len(self.images) == batch_size:
                inputs = np.array(self.images)
                targets = np.array(self.labels)
                self.reset()
                yield inputs, targets
                
class ValidationGenerator(object):
    def __init__(self, valD, valE, valP):
        self.reset()
        self.valD=valD
        self.valE=valE
        self.valP=valP

    def reset(self):
        self.images = []
        self.labels = []
        self.l=0
    
    def flow_from_directory(self, batch_size=36):
        # LabelEncode(classをint型に変換)するためのdict
            # ディレクトリから画像のパスを取り出す
        while(1>0):
            k=random.randint(0,2)
            if k==0:
                m=random.randint(0,len(self.valD)-1)
                img=image.load_img(self.valD[m],target_size=(240,320,3))
                img =image.img_to_array(img)
                img=img/255
                self.images.append(img)
                self.labels.append([1,0,0])

            if k==1:
                m=random.randint(0,len(self.valE)-1)
                img=image.load_img(self.valE[m],target_size=(240,320,3))
                img=image.img_to_array(img)
                img=img/255
                self.images.append(img)
                self.labels.append([0,1,0])

            if k==2:
                m=random.randint(0,len(self.valP)-1)
                img=image.load_img(self.valP[m],target_size=(240,320,3))
                img=image.img_to_array(img)
                img=img/255
                self.images.append(img)
                self.labels.append([0,0,1])

            # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納
            # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする
            if len(self.images) == batch_size:
                inputs = np.array(self.images)
                targets = np.array(self.labels)
                self.reset()
                yield inputs, targets
