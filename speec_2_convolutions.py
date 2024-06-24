import tensorflow as tf
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM,BatchNormalization
from keras._tf_keras.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from preproces import *
from scipy.io import wavfile


# Save data to array file first
max_len = 11
buckets = 20
save_data_to_array(max_len=max_len, n_mfcc=buckets)
labels=["bed", "bird", "cat" , "dog","down","eight","five","four","go","happy","house","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","wow","yes","zero"]

# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()
channels = 1
epochs = 50
batch_size = 100
num_classes = 30

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)
background_noises = []
background_noise_path = './_background_noise_'  # Replace with the actual path to your background noise folder
for noise_file in os.listdir(background_noise_path):
    if noise_file.endswith('.wav'):
        rate, data = wavfile.read(os.path.join(background_noise_path, noise_file))
        background_noises.append(data)

def add_background_noise(audio, noise_data, noise_level=0.0009):
    noise = noise_data[np.random.randint(len(noise_data))]
    start = np.random.randint(0, len(noise) - len(audio))
    noise_segment = noise[start:start + len(audio)]
    augmented_audio = audio + noise_level * noise_segment
    augmented_audio = augmented_audio.astype(np.int16)
    return augmented_audio

# Add noise to training data
for i in range(len(X_train)):
    X_train[i] = add_background_noise(X_train[i].flatten(), background_noises).reshape(buckets, max_len, channels)

print(X_train[100])

plt.imshow(X_train[50, :, :, 0])
plt.show()
print(y_train[50])

y_train_hot = to_categorical(y_train, num_classes)
y_test_hot = to_categorical(y_test, num_classes)

# Improved CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(buckets, max_len, channels), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(X_train, y_train_hot, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_hot), callbacks=[tensorboard_callback])

score = model.evaluate(X_test, y_test_hot, verbose=0)
print('\n', 'Test accuracy:', score[1])

# ذخیره مدل
model.save('./models/3_convolutions(new).h5')
