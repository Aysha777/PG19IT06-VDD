from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
print(tf.__version__)
import json
import numpy as np
import pandas as pd
import os
import csv
from random import shuffle

uuid = []
with open("data1.csv", 'r') as file:
   reader = csv.reader(file)  
   for row in reader:
      uuid.append(row[0])

for i in uuid:
   filepath = os.path.join("org/datasets/",  i , "/data")


dataset = []


X = []
y = []

for person in dataset:
    spectogram = person['spectogram']/np.float32(255) #normalize input pixels 
    status = int(person['status'])
    X.append(spectogram)
    y.append(status)
X = np.array(X)
y = np.array(y)

X=X.reshape((-1, 1, 28, 28))
print('X shape: ', X.shape, 'y shape: ', y.shape)
