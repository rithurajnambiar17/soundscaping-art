from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import librosa

def extract_feature_and_print_prediction(file_name):
    encoder = LabelEncoder()
    encoder.classes_ = np.load('le.npy')
    modelFile = tf.keras.models.load_model('model.h5')
    audio_data, sample_rate = librosa.load(file_name) 
    fea = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50)
    scaled = np.mean(fea.T,axis=0)
    pred_fea = np.array([scaled])
    pred_vector = np.argmax(modelFile.predict(pred_fea),axis=-1)
    pred_class = encoder.inverse_transform(pred_vector)
    # print("The Predicted class is:", pred_class[0], '\n')
    return pred_class[0]