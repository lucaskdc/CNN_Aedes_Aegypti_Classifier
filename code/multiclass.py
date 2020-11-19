import csv

import glob
import os
import matplotlib.pyplot as plt

import librosa
import numpy as np
from keras import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm

def windows(data, window_size):
  start = 0
  while start < len(data):
    yield start, start + window_size
    start += (window_size // 2)

def extract_features(sub_dirs, file_ext="*.wav"):
  window_size = hop_length * (frames - 1)
  log_specgrams = []
  labels = []
  for l, sub_dir in enumerate(sub_dirs):
    for fn in glob.glob(os.path.join(file_url, sub_dir, file_ext)):
      sound_clip, _ = librosa.load(fn, sr=sample_rate)
      print('Extracting features from: ' + fn)
      label = fn.split('\\')[-2]
      for (start, end) in windows(sound_clip, window_size):
        if (len(sound_clip[start:end]) == window_size):
          signal = sound_clip[start:end]
          melspec = librosa.feature.melspectrogram(signal, n_mels=bands, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
          logspec = librosa.power_to_db(melspec, ref=np.max)
          logspec = logspec / 80 + 1
          logspec = logspec.T.flatten()[:, np.newaxis].T
          log_specgrams.append(logspec)
          labels.append(label)
  features = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)

  return np.array(features), np.array(labels, dtype=np.int)

def load_data():
  tr_sub_dirs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]
  tr_features, tr_labels = extract_features(tr_sub_dirs)
  np.savez(file_url, tr_features, tr_labels)
  return tr_features, tr_labels
  # Comment the above code and use the code below to not process the files again
  npread = np.load(file_url + '.npz')
  return npread['arr_0'], npread['arr_1']

def create_model():
  model = Sequential()

  model.add(Conv2D(32, (20, 5), activation='relu', input_shape=(bands, frames, 1), padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(32, (8, 4), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(23, activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.summary()
  return model

def train_and_evaluate_model(model, xtrain, ytrain, xval, yval):
  model.fit(xtrain, ytrain,  batch_size=32, epochs=10, verbose=2)
  y_predicted_classes = model.predict_classes(xval)
  conf_matrix = skm.confusion_matrix(yval, y_predicted_classes)
  print("Confusion matrix: \n"  + str(conf_matrix))
  score = model.evaluate(xval, yval, verbose=0)
  print("General accuracy: %.2f%%" % ( score[1] * 100))

  accuracy = skm.accuracy_score(yval, y_predicted_classes)
  print("Accuracy: ")
  print(accuracy)

  precision = skm.precision_score(yval, y_predicted_classes, average=None)
  print("Precision: ")
  print(precision)

  recall = skm.recall_score(yval, y_predicted_classes, average=None)
  print("Recall: ")
  print(recall)

  f1_score = skm.f1_score(yval, y_predicted_classes, average=None)
  print("F1-score: ")
  print(f1_score)

  return score[1], precision, recall, f1_score, conf_matrix

seed = 123
np.random.seed(seed)  # for reproducibility

file_url = 'E:\\mosquitos\\train'

bands = 60
frames = 60
hop_length = 128
n_fft = 1024

sample_rate = 8000

n_folds = 10
X_train, Y_train = load_data()

skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
conf_matrixes = []

for i, (train, test) in enumerate(skf.split(X_train, Y_train)):
  print("Running Fold", i+1, "/", n_folds)
  model = None # Clearing the NN.
  model = create_model()

  # Generate batches from indices
  xtrain, xval = X_train[train], X_train[test]
  ytrain, yval = Y_train[train], Y_train[test]
  accuracy, precision, recall, f1_score, conf_matrix = train_and_evaluate_model(model, xtrain, ytrain, xval, yval)
  accuracy_scores.append(accuracy)
  precision_scores.append(precision)
  recall_scores.append(recall)
  f1_scores.append(f1_score)
  conf_matrixes.append(conf_matrix)


print("Accuracy scores: " + str(accuracy_scores))
print("Precision scores: " + str(precision_scores))
print("Recall scores: " + str(recall_scores))
print("F1 scores: " + str(f1_scores))

mean_confusion_matrix = np.mean(np.array(conf_matrixes), axis=0)
print("Average of confusion matrix: " + repr(mean_confusion_matrix))

csv_filename = 'multiclass.csv'

with open(csv_filename, 'w', newline='') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(['specie', 'fold', 'precision', 'recall', 'f1_score'])

  for specie in range(23):
    for fold in range(10):
      csv_writer.writerow([specie, fold + 1, precision_scores[fold][specie], recall_scores[fold][specie], f1_scores[fold][specie]])