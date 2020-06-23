import keras
import numpy as np
from keras.layers import Conv2D,Input, MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten
import tensorflow as tf
import os
from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras.initializers import GlorotNormal as GN
from tensorflow.keras.initializers import GlorotUniform as GU

MEAN = 0.0
STDDEV = 0.01
SEED = 5
class VGG_block(keras.layers.Layer):
  def __init__(self, input_shape=(224,224,3)):
    super(VGG_block, self).__init__()
    self.conv1_1 = Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv1_2 = Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.max_pooling1 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv2_1 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv2_2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.max_pooling2 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv3_1 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv3_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv3_3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.max_pooling3 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv4_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv4_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv4_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.max_pooling4 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv5_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv5_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.conv5_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=GU(SEED))
    self.max_pooling5 = MaxPool2D(pool_size=(2,2),strides=(2,2))
    # self.avg_pooling = AveragePooling2D(pool_size=(7, 7), padding="same")
    # self.model = keras.Model(inputs= self.conv1_1, outputs=self.max_pooling5)
  
  # @tf.function
  def call(self, inputs):
    x = self.conv1_1(inputs)
    x = self.conv1_2(x)
    x = self.max_pooling1(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.max_pooling2(x)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.conv3_3(x)
    x = self.max_pooling3(x)
    x = self.conv4_1(x)
    x = self.conv4_2(x)
    x = self.conv4_3(x)
    x = self.max_pooling4(x)
    x = self.conv5_1(x)
    x = self.conv5_2(x)
    x = self.conv5_3(x)
    x = self.max_pooling5(x)
    return x





class MRNet_vgg_layer(keras.layers.Layer):
  def __init__(self, input_shape, batch_size):
    super(MRNet_vgg_layer, self).__init__()
    # (1, s, 224, 224, 3)
    self.vgg = VGG_block(input_shape=input_shape[2:])
    #(s, 7, 7, 512)
    self.avg_pooling = AveragePooling2D(pool_size=(7, 7), padding="same")
    #(s, 1, 1, 512)
    self.dropout = Dropout(0.5)
    self.fc = Dense(1, activation="sigmoid", input_dim=512, kernel_initializer=GU(SEED))
    self.b_size = batch_size
    

  @tf.function(autograph=True)
  def call(self, inputs):
    arr = []
    for index in range(self.b_size):
      out = self.vgg(inputs[index])
      out = tf.squeeze(self.avg_pooling(out), axis=[1, 2])
      #(s, 512)
      out = keras.backend.max(out, axis=0, keepdims=True)
      #(1, 512)
      out = tf.squeeze(out)
      #(512)
      arr.append(out) 
    output = tf.stack(arr, axis=0)
    #(1, 512)
    output = self.fc(self.dropout(output))
    #(1, 1)
    return output

  def compute_output_shape(self, input_shape):
    return (None, 1)


def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

def MRNet_vgg_model(batch_size, lr, combination = ["abnormal", "axial"]):
  METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
  ]

  b_size = batch_size
  model = keras.Sequential()
  model.add(MRNet_vgg_layer((None, None, 224, 224, 3), b_size))
  model(Input(shape=(None, 224, 224, 3)))
  model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=lr),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=METRICS)
      
  data_path = "/content/gdrive/My Drive/Colab Notebooks/MRNet/"
  checkpoint_dir = data_path+"training_vgg/" + combination[0] + "/" + combination[1] + "/"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"weights.{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 verbose=1)
  tcb = TestCallback(model)
  return model, [cp_callback, tcb]

class TestCallback(tf.keras.callbacks.Callback):
  def __init__(self, model):
    super(TestCallback, self).__init__()
    self.model = model

  def on_epoch_end(self, epoch, logs=None):
    if(epoch == 0):
      self.w = self.model.layers[0].get_weights()[0]
      return
    self.w_after = self.model.layers[0].get_weights()[0]
    print('  TestCallback: ', (self.w == self.w_after).all())
    self.w = self.w_after










    