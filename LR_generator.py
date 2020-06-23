import keras
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random, copy

MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class MRNet_LR_data_generator(keras.utils.Sequence):
  def __init__(self, datapath, IDs, labels, models, shuffle=True,
               scale_to = (256, 256), label_type="abnormal",
               data_type='train', model="vgg"):
    print("Initializing LR Data Generator:")
    self.path = datapath
    self.n = 0
    self.IDs = IDs
    self.labels = labels
    self.shuffle = shuffle
    self.scale_to = scale_to
    self.label_type = label_type
    self.exam_types = ["axial", "coronal", "sagittal"]
    self.models = models
    self.model = model
    self.data_type = data_type
    self.data_path = os.path.join(self.path, self.data_type)
    self.data = []
    self.end = self.__len__()
    self.predict_all()
    self.on_epoch_end()
    

  def predict_all(self):
    print("loading data..")
    bar = tf.keras.utils.Progbar(self.end)
    for i in range(self.end):
      pair = self.get(i)
      self.data.append(pair)
      bar.update(i)
    bar.update(self.end, finalize=True)

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.IDs[self.data_type][self.exam_types[0]]))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __data_generation(self, ID):
    'Generates data containing batch_size samples' 
    y = np.empty((1,), dtype=int)
    X = []
    for t in self.exam_types:
      exam_path = os.path.join(self.data_path, t)
      exam = np.load(os.path.join(exam_path, ID+'.npy'))
      e = []
      for s in exam:
        im = Image.fromarray(s)
        s = np.array(im.resize(self.scale_to), dtype=np.float32)
        s = (s - np.min(s)) / (np.max(s) - np.min(s)) * MAX_PIXEL_VAL
        s = (s - MEAN) / STDDEV
        expanded = np.array([s])
        e.append(expanded.reshape((self.scale_to[0], self.scale_to[1], 1)))

      e = np.array(e)
      X.append(e)
    y[0] = self.labels[ID][self.label_type]        
    return X, y

  def __len__(self):
    'Denotes the number of batches per epoch'
    IDs_len = len(self.IDs[self.data_type][self.exam_types[0]])
    return int(IDs_len)

  def get(self, index):
    'Generate one batch of data'
    X, y = self.__data_generation(self.IDs[self.data_type][self.exam_types[0]][index])
    X = self.process_data(X)
    return X, y
  def __getitem__(self, index):
    index = self.indexes[index]
    return self.data[index]

  def process_data(self, exam):
    y_score = []
    for i in range(3):
      e = []
      for s in range(0, exam[i].shape[0]):
        scan = exam[i][s]
        scan = scan.reshape((self.scale_to[0], self.scale_to[1]))
        scan = np.array([scan, scan, scan]).reshape((self.scale_to[0], self.scale_to[1], 3))
        e.append(scan)
      e = np.array(e)
      e = e.reshape((1, exam[i].shape[0], self.scale_to[0], self.scale_to[1], 3))
      p = self.models[i].predict_proba(e, batch_size=1)
      y_score.append(p)
    y = np.array(y_score)
    y = y.reshape(1, 3)
    return y

  def __next__(self):
    if self.n >= self.end:
      self.n = 0
    result = self.__getitem__(self.n)
    self.n += 1
    return result
