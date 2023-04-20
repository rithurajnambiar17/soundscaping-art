###############################
# Importing Necessary Libraries
###############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import librosa.display
import tensorflow as tf
import IPython.display as ipd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tqdm.auto import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation , Dropout

################################
# Analysing Data Type and Format
################################

df = pd.read_csv("D:/Coding/Python/PROJECTS/Stability/Sound/UrbanSound8K.csv")
print(df.head())

############################################################
# Using Librosa to analyse random sound sample - SPECTROGRAM
############################################################

dat1, sampling_rate1 = librosa.load('D:/Coding/Python/PROJECTS/Stability/Sound/fold5/100032-3-0-0.wav')
dat2, sampling_rate2 = librosa.load('D:/Coding/Python/PROJECTS/Stability/Sound/fold5/100263-2-0-117.wav')

######
# dat1
######

plt.figure(figsize=(20, 20))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()

######
# dat2
######

plt.figure(figsize=(20, 20))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()


arr = np.array(df["slice_file_name"])
fold = np.array(df["fold"])
cla = np.array(df["class"])

for i in range(192, 197, 2):
    path = 'D:/Coding/Python/PROJECTS/Stability/Sound/fold' + str(fold[i]) + '/' + arr[i]
    data, sampling_rate = librosa.load(path)
    plt.figure(figsize=(10, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(cla[i])
    plt.show()

"""
Feature Extraction and Database Building

• Mel Spectrograms
• MFCC
"""

def features_extract(file):
    sample,sample_rate = librosa.load(file_name)
    feature = librosa.feature.mfcc(y=sample,sr=sample_rate,n_mfcc=50)
    scaled_feature = np.mean(feature.T,axis=0)
    return scaled_feature

extracted = []
path = 'D:/Coding/Python/PROJECTS/Stability/Sound'

for index_num,row in tqdm(df.iterrows()):
    file_name = os.path.join(path +'/'+'fold'+ str(row["fold"])+'/', str(row['slice_file_name'])).replace("\\","/")
    final_class_labels = row['class']
    data = features_extract(file_name)
    extracted.append([data,final_class_labels])

ext_df = pd.DataFrame(extracted,columns=['feature','class'])
print(ext_df)

x = np.array(ext_df['feature'].tolist())
y = np.array(ext_df['class'].tolist())

le = LabelEncoder()
y = to_categorical(le.fit_transform(y))


####################
# Train - Test split
####################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
print("Number of training samples = ", x_train.shape[0])
print("Number of testing samples = ",x_test.shape[0])

###########################
# Artificial Neural Network
###########################

num_labels = y.shape[1]
model = Sequential()

model.add(Dense(128, input_shape=(50,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))

model.add(Dense(num_labels))
model.add(Activation('softmax'))
print(model.summary())


#####################
# Compiling the model
#####################

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

###################
# Fitting the model
###################

model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

####################################
# Extracting features for prediction
####################################

def extract_feature_and_print_prediction(file_name):
    audio_data, sample_rate = librosa.load(file_name) 
    fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    scaled = np.mean(fea.T,axis=0)
    pred_fea = np.array([scaled])
    pred_vector = np.argmax(model.predict(pred_fea),axis=-1)
    pred_class = le.inverse_transform(pred_vector)
    print("The Predicted class is:", pred_class[0], '\n')

###################
# Testing the model
###################

extract_feature_and_print_prediction('D:/Coding/Python/PROJECTS/Stability/Sound/fold2/100652-3-0-2.wav')
ipd.Audio('D:/Coding/Python/PROJECTS/Stability/Sound/fold2/100652-3-0-2.wav')

############################
# Making a h5 file to upload
############################

import joblib
filename = 'sound_classification.h5'
joblib.dump(extract_feature_and_print_prediction, filename)