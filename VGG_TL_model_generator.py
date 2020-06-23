import keras
import numpy as np
from keras.layers import Conv2D,Input, MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten
from keras.applications import VGG16
import tensorflow as tf
import os
from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras.initializers import GlorotNormal as GN
from tensorflow.keras.initializers import GlorotUniform as GU

MEAN = 0.0
STDDEV = 0.01
SEED = 5

class MRNet_vgg_layer(keras.layers.Layer):
  def __init__(self, input_shape, batch_size):
    super(MRNet_vgg_layer, self).__init__()
    self.vgg = VGG16(
      include_top=False,
      weights="imagenet",
      input_tensor=None,
      input_shape=(224, 224, 3),
      pooling="avg",
      classes=1,
    )
    # self.avg_pooling = AveragePooling2D(pool_size=(7, 7), padding="same")
    self.dropout = Dropout(0.5)
    self.fc = Dense(1, activation="sigmoid", input_dim=512, kernel_initializer=GU(SEED))
    self.b_size = batch_size
    

  @tf.function(autograph=True)
  def call(self, inputs):
    arr = []
    for index in range(self.b_size):
      out = self.vgg(inputs[index])
      # out = tf.squeeze(self.avg_pooling(out), axis=[1, 2])
      out = keras.backend.max(out, axis=0, keepdims=True)
      out = tf.squeeze(out)
      arr.append(out) 
    output = tf.stack(arr, axis=0)
    output = self.fc(self.dropout(output))
    return output

  def compute_output_shape(self, input_shape):
    return (None, 1)


def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

def MRNet_vgg_tl_model(batch_size, lr, combination = ["abnormal", "axial"]):
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
      optimizer=tf.keras.optimizers.Adam(lr=lr, decay=0.005),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=METRICS)
  data_path = "/content/gdrive/My Drive/Colab Notebooks/MRNet/"
  checkpoint_dir = data_path+"training_vgg_TL/" + combination[0] + "/" + combination[1] + "/"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"weights.{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 verbose=1)
  # tcb = TestCallback(model)
  return model, [cp_callback]

class validation_Callback(tf.keras.callbacks.Callback):
  def __init__(self, model, valid_data_gen):
    super(validation_Callback, self).__init__()
    self.model = model
    self.valid_data_gen = valid_data_gen

  def on_epoch_end(self, epoch, logs=None):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    y_true = []
    y_score = []
    for i in range(len(self.valid_data_gen)):
      x, y = next(self.valid_data_gen)
      _y = self.model.predict_classes(x, batch_size=1, verbose=0)
      # _y_score = self.model.predict_proba(x, batch_size=1)
      # y_score.append(_y_score[0][0])
      y_true.append(y[0])
      if _y[0][0] == 1:
        if _y[0][0] == y[0]:
          tp += 1
        else:
          fp += 1
      else:
        if _y[0][0] == y[0]:
          tn += 1
        else:
          fn += 1
    # y_score = np.array(y_score)
    y_true = np.array(y_true)
    # fpr, tpr, _ = roc_curve(y_true, y_score)
    # auc = roc_auc(fpr, tpr)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*((precision*recall)/(precision+recall))
    print ("\n validation: tp = ", tp, " fp = ", fp, " tn = ", tn, " fn = ", fn, " accuracy = ", (tp+tn)/(len(self.valid_data_gen)), "F1 Score = ", f1, "\n")

    










    