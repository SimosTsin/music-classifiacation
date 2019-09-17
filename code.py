# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:02:48 2019

@author: Simos
"""

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pathlib
import csv


import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir('path'):
        songname = 'path'/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=512, Fs=0.5, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()

## Features extraction ##
header = 'filename mean_chroma_stft min_chroma_stft max_chroma_stft std_chroma_stft \
 mean_rmse min_rmse max_rmse std_rmse mean_spectral_centroid min_spectral_centroid max_spectral_centroid std_spectral_centroid \
 mean_spectral_bandwidth min_spectral_bandwidth max_spectral_bandwidth std_spectral_bandwidth \
 mean_rolloff_80 min_rolloff_80 max_rolloff_80 std_rolloff_80 \
 mean_rolloff_85 min_rolloff_85 max_rolloff_85 std_rolloff_85 \
 mean_rolloff_90 min_rolloff_90 max_rolloff_90 std_rolloff_90 \
 mean_rolloff_95 min_rolloff_95 max_rolloff_95 std_rolloff_95 \
 mean_rolloff_99 min_rolloff_99 max_rolloff_99 std_rolloff_99 \
 mean_zero_crossing_rate min_zero_crossing_rate max_zero_crossing_rate std_zero_crossing_rate \
 mean_spectral_flatness min_spectral_flatness max_spectral_flatness std_spectral_flatness \
 mean_poly_features min_poly_features max_poly_features std_poly_features'
for i in range(1, 21):
    header += f' mean_mfcc{i} min_mfcc{i} max_mfcc{i} std_mfcc{i}'
header += ' label'
header = header.split()

file = open('data_plus.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
for g in genres:
    for filename in os.listdir('path'):
        songname = 'path'/{filename}'
        y, sr = librosa.core.load(songname, mono=True, sr=44100)
        seconds_per_split = 0.5 
        samples_per_split = librosa.core.time_to_samples(seconds_per_split, sr=sr) 
        total_splits = len(y) / samples_per_split 
        hop_length = samples_per_split 
        split_times = np.array([seconds_per_split * i for i in range(0, int(total_splits)+1)])
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        rmse = librosa.feature.rmse(y=y, hop_length=hop_length)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
        rolloff_80 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.8)
        rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)
        rolloff_90 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.9)
        rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.95)
        rolloff_99 = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.99)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
        flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_length) 
        poly_f = librosa.feature.poly_features(y=y, sr=sr, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.min(chroma_stft)} {np.max(chroma_stft)} {np.std(chroma_stft)} \
         {np.mean(rmse)} {np.min(rmse)} {np.max(rmse)} {np.std(rmse)} \
         {np.mean(spec_cent)} {np.min(spec_cent)} {np.max(spec_cent)} {np.std(spec_cent)} \
         {np.mean(spec_bw)} {np.min(spec_bw)} {np.max(spec_bw)} {np.std(spec_bw)} \
         {np.mean(rolloff_80)} {np.min(rolloff_80)} {np.max(rolloff_80)} {np.std(rolloff_80)} \
         {np.mean(rolloff_85)} {np.min(rolloff_85)} {np.max(rolloff_85)} {np.std(rolloff_85)} \
         {np.mean(rolloff_90)} {np.min(rolloff_90)} {np.max(rolloff_90)} {np.std(rolloff_90)} \
         {np.mean(rolloff_95)} {np.min(rolloff_95)} {np.max(rolloff_95)} {np.std(rolloff_95)} \
         {np.mean(rolloff_99)} {np.min(rolloff_99)} {np.max(rolloff_99)} {np.std(rolloff_99)} \
         {np.mean(zcr)} {np.min(zcr)} {np.max(zcr)} {np.std(zcr)} \
         {np.mean(flat)} {np.min(flat)} {np.max(flat)} {np.std(flat)} \
         {np.mean(poly_f)} {np.min(poly_f)} {np.max(poly_f)} {np.std(poly_f)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)} {np.min(e)} {np.max(e)} {np.std(e)}'
        to_append += f' {g}'
        file = open('data_plus.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

## Data proccessing ##
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Correlation matrix #

data = pd.read_csv('data_plus.csv')
data = data.drop(['filename'],axis=1)

plt.matshow(data.corr())

data.head()
data.shape
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(y_train)
len(y_test)
X_train[10]

## Model construction ##
import keras
from keras import models
from keras import layers
from keras.utils.vis_utils import plot_model

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train,
                    y_train,
                    epochs=40,
                    batch_size=512)

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

test_loss, test_acc= model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)
print('test_loss: ',test_loss)
predictions = model.predict(X_test)

## Confusion matrix and Classification report ##
from sklearn.metrics import classification_report, confusion_matrix
y_pred = np.argmax(predictions, axis=1)

y_test_string = []
for l in y_test.tolist():
    if l == 0:
        y_test_string.append(genres[0])
    elif l == 1:
        y_test_string.append(genres[1])
    elif l == 2:
        y_test_string.append(genres[2])
    elif l == 3:
        y_test_string.append(genres[3])
    elif l == 4:
        y_test_string.append(genres[4])
    elif l == 5:
        y_test_string.append(genres[5])
    elif l == 6:
        y_test_string.append(genres[6])
    elif l == 7:
        y_test_string.append(genres[7])
    elif l == 8:
        y_test_string.append(genres[8])
    elif l == 9:
        y_test_string.append(genres[9])


y_pred_string = []       
for l in y_pred.tolist():
    if l == 0:
        y_pred_string.append(genres[0])
    elif l == 1:
        y_pred_string.append(genres[1])
    elif l == 2:
        y_pred_string.append(genres[2])
    elif l == 3:
        y_pred_string.append(genres[3])
    elif l == 4:
        y_pred_string.append(genres[4])
    elif l == 5:
        y_pred_string.append(genres[5])
    elif l == 6:
        y_pred_string.append(genres[6])
    elif l == 7:
        y_pred_string.append(genres[7])
    elif l == 8:
        y_pred_string.append(genres[8])
    elif l == 9:
        y_pred_string.append(genres[9])
        
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    
cm_analysis(y_test_string, y_pred_string, 'conf_matrix_b', genres, ymap=None, figsize=(10,10))

print(classification_report(y_test, y_pred, target_names=genres))
