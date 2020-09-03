#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import sys
import copy
import string
import os
from sklearn.preprocessing import MinMaxScaler
import operator

path = "books/"

data = os.listdir(path)
ans = open("CORPUS.txt", 'w')
for f in data:
    with open(path + f, 'r', 
              encoding='ascii', errors='ignore') as fileDataStream:
        for line in fileDataStream:
            ans.write(line)

ans.close()
ans = ans.name

contents = open(ans, 'r').read()

text = copy.copy(contents)
char = set(contents)
text = contents.lower()
char = set(contents.lower())
char = char.difference(set(string.punctuation))
text = text.translate(str.maketrans('', '', string.punctuation))

processed_text = text
CharacterSet = char

char_to_Ascii = dict()
for index, char in enumerate(sorted(CharacterSet)):
    char_to_Ascii[char] = index


dict_scaled_char = dict()
scaledValues = MinMaxScaler().fit_transform(np.array(list(char_to_Ascii.values())).reshape(-1, 1))
for index in range(len(scaledValues)):
    dict_scaled_char[list(char_to_Ascii.keys())[index]] = scaledValues[index][0]



ascii_to_Char = {v:k for k, v in char_to_Ascii.items()}


window = 99
X_data = []
Y_data = []
for i in range(0, len(processed_text) - window, 1):
    temp = processed_text[i:i + window]
    last = processed_text[i + window]
    X_data.append([dict_scaled_char[char] for char in temp])
    Y_data.append(char_to_Ascii[last])

numBlocks = len(X_data)
X_data = np.reshape(X_data, (numBlocks, window, 1))
Y_data = keras.utils.to_categorical(Y_data)


mem_unit = 256

model = keras.models.Sequential([
    keras.layers.LSTM(units=mem_unit,
                      input_shape=(X_data.shape[1], X_data.shape[2])),
    keras.layers.Dense(Y_data.shape[1], activation="softmax"),
])
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))
model.summary()



epochs = 30
output_dir = "./text_generation"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
checkpoint_prefix = os.path.join(output_dir, 'ck_{epoch:02d}.hdf5')
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    monitor='loss',
    save_weights_only= True,
    mode='min'
)
earlyStoppingCk = keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-4)
history = model.fit(X_data, Y_data, epochs=epochs, batch_size=128, 
                    callbacks=[checkpoint_callback, earlyStoppingCk])

min_loss_point = output_dir +'/ck_30.hdf5'
model.load_weights(min_loss_point)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))

test_tex= 'There are those who take mental phenomena naively, just as they would physical phenomena. This school of psychologists tends not to emphasize the object.'
test_tex= init_text.translate(str.maketrans('', '', string.punctuation))

generated_text = copy.copy(init_text.lower())
encodeList = [dict_scaled_char[char] for char in generated_text][-99:]
for _ in range(1000):
    data = np.reshape(encodeList, (1, len(encodeList), 1))
    pred = model.predict(data)
    charIndex = np.argmax(pred)
    char = ascii_to_Char[np.argmax(pred)]
    generated_text += char
    encodeList.append(dict_scaled_char[char])
    encodeList = encodeList[1:len(encodeList)]
    
print(generated_text)

