import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import load_model
from preproces import *  # فرض می‌کنیم که این تابع برای محاسبه ویژگی‌ها (مانند MFCC) وجود دارد
import librosa
import matplotlib.pyplot as plt

# تابع برای بارگذاری مدل و انجام پیش‌بینی
def predict_audio(model_path, audio_path, max_len, buckets):
    # بارگذاری مدل
    model = load_model(model_path)
    
    # پیش‌پردازش نمونه صدا
    mfcc_features = wav2mfcc(audio_path, n_mfcc=buckets, max_len=max_len)
    mfcc_features = mfcc_features.reshape(1, buckets, max_len, 1)
    class_names=["bed", "bird", "cat" , "dog","down","eight","five","four","go","happy","house","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","wow","yes","zero"]

    # پیش‌بینی
    prediction = model.predict(mfcc_features)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_class = class_names[predicted_label[0]]
    
    return predicted_label, prediction,predicted_class

# مسیر مدل آموزش‌دیده
model_path = './models/3_convolutions(new).h5'

# مسیر نمونه صدا
audio_path = './new_test/clip_000c2c07b.wav'

# مقدار max_len و buckets باید مطابق با مقادیری باشد که در حین آموزش مدل استفاده شده است
max_len = 11
buckets = 20

predicted_label, prediction,predicted_class = predict_audio(model_path, audio_path, max_len, buckets)

#print("Predicted label:", predicted_label)
print("Predicted class:", predicted_class)
#print("Prediction probabilities:", prediction)
