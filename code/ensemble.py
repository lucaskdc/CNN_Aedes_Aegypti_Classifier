import csv
import glob
import os
import matplotlib.pyplot as plt

import librosa
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm


def windows(data, window_size):
  start = 0
  while start < len(data):
    yield start, start + window_size
    start += (window_size // 2)

def extract_features( sub_dirs, file_ext="*.wav"):
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
  np_labels = np.array(labels, dtype=np.int)

  print("Loading models")
  aedes_albopictus = predict_cnn('aedes_albopictus', features)
  aedes_mediovittatus = predict_cnn('aedes_mediovittatus', features)
  aedes_sierrensis = predict_cnn('aedes_sierrensis', features)
  anopheles_albimanus = predict_cnn('anopheles_albimanus', features)
  anopheles_arabiensis_dongola = predict_cnn('anopheles_arabiensis_dongola', features)
  anopheles_arabiensis_rufisque = predict_cnn('anopheles_arabiensis_rufisque', features)
  anopheles_atroparvus = predict_cnn('anopheles_atroparvus', features)
  anopheles_dirus = predict_cnn('anopheles_dirus', features)
  anopheles_farauti = predict_cnn('anopheles_farauti', features)
  anopheles_freeborni = predict_cnn('anopheles_freeborni', features)
  anopheles_gambiae_akron = predict_cnn('anopheles_gambiae_akron', features)
  anopheles_gambiae_kisumu = predict_cnn('anopheles_gambiae_kisumu', features)
  anopheles_gambiae_rsp = predict_cnn('anopheles_gambiae_rsp', features)
  anopheles_merus = predict_cnn('anopheles_merus', features)
  anopheles_minimus = predict_cnn('anopheles_minimus', features)
  anopheles_quadriannulatus = predict_cnn('anopheles_quadriannulatus', features)
  anopheles_quadrimaculatus = predict_cnn('anopheles_quadrimaculatus', features)
  anopheles_stephensi = predict_cnn('anopheles_stephensi', features)
  culex_pipiens = predict_cnn('culex_pipiens', features)
  culex_quinquefasciatus = predict_cnn('culex_quinquefasciatus', features)
  culex_tarsalis = predict_cnn('culex_tarsalis', features)
  culiseta_incidens = predict_cnn('culiseta_incidens', features)
  input = np.stack((aedes_albopictus, aedes_mediovittatus, aedes_sierrensis, anopheles_albimanus, anopheles_arabiensis_dongola, anopheles_arabiensis_rufisque,
                      anopheles_atroparvus, anopheles_dirus, anopheles_farauti, anopheles_freeborni, anopheles_gambiae_akron, anopheles_gambiae_kisumu, anopheles_gambiae_rsp,
                      anopheles_merus, anopheles_minimus, anopheles_quadriannulatus, anopheles_quadrimaculatus, anopheles_stephensi, culex_pipiens, culex_quinquefasciatus, culex_tarsalis,
                      culiseta_incidens), axis=1)
  return input, np_labels

def load_data():
  tr_sub_dirs = ["0", "1"]
  tr_features, tr_labels = extract_features(tr_sub_dirs)
  return tr_features, tr_labels

def predict_cnn(model_name, features):
  file_rul = 'binaries_models/' + model_name
  json_file = open(file_rul + '.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(file_rul + '.h5')

  loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  predicted = loaded_model.predict_classes(features)
  return predicted

def train_and_evaluate_model(xtrain, ytrain, xval, yval):
  model = Sequential()
  model.add(Dense(3, input_dim=22, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  history = model.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=32, epochs=10, verbose=2)

  y_predicted_classes = model.predict_classes(xval)
  conf_matrix = skm.confusion_matrix(yval, y_predicted_classes, labels=[1, 0])
  print("Confusion matrix: \n"  + str(conf_matrix))
  score = model.evaluate(xval, yval, verbose=0)
  print("Accuracy: %.2f%%" % ( score[1] * 100))
  precision = skm.precision_score(yval, y_predicted_classes, pos_label=0)
  print("Precision: %.2f%%" % (precision*100))
  recall = skm.recall_score(yval, y_predicted_classes, pos_label=0)
  print("Recall: %.2f%%" % (recall*100))
  f1_score = skm.f1_score(yval, y_predicted_classes, pos_label=0)
  print("F1-score: %.4f" % f1_score)
  auroc = skm.roc_auc_score(yval, y_predicted_classes)
  print("AUROC: %.4f" % auroc)

  return score[1], precision, recall, f1_score

seed = 123
np.random.seed(seed)  # for reproducibility

file_url = 'E:\\mosquitos\\train'

bands = 60
frames = 40
hop_length=256
n_fft=1024

sample_rate = 8000

n_folds = 10
X, Y = load_data()

skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for i, (train, val) in enumerate(skf.split(X, Y)):
  print("Running Fold", i+1, "/", n_folds)

  # Generate batches from indices
  xtrain, xval = X[train], X[val]
  ytrain, yval = Y[train], Y[val]
  accuracy, precision, recall, f1 = train_and_evaluate_model(xtrain, ytrain, xval, yval)
  accuracy_scores.append(accuracy)
  precision_scores.append(precision)
  recall_scores.append(recall)
  f1_scores.append(f1)

print("Accuracy scores: " + str(accuracy_scores))
print("Precision scores: " + str(precision_scores))
print("Recall scores: " + str(recall_scores))
print("F1 scores: " + str(f1_scores))

csv_filename = "ensemble_neural_network.csv"

with open(csv_filename, 'w', newline='') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(['accuracy', 'precision', 'recall', 'f1_score'])

  for i in range(len(accuracy_scores)):
    csv_writer.writerow([accuracy_scores[i], precision_scores[i], recall_scores[i], f1_scores[i]])