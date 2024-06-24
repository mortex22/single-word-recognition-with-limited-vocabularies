# single-word-recognition-with-limited-vocabularies
Keyword Recognition and Classification Using MFCC Features and Convolutional Neural Networks from Audio Files

1. Preproccess.py:
This script processes audio files for use in machine learning tasks. It includes functions to load audio data, convert it into Mel-frequency cepstral coefficients (MFCC) features, save these features, and prepare the dataset for training and testing. Additionally, it provides a function to visualize the audio spectrogram.

Prerequisites
Make sure to install the necessary Python libraries before running the script:

bash
Copy code
pip install librosa scikit-learn keras numpy tqdm matplotlib
Directory Structure
The script expects the audio files to be organized in a directory where each subdirectory represents a label (or class). For example:

Copy code
new_train/
├── label1/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── label2/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── ...
Functions
* get_labels(path=DATA_PATH)
Input: Directory path containing labeled subdirectories.
Output: Tuple containing labels, label indices, and one-hot encoded labels.
Description: Reads the subdirectory names (labels) and generates corresponding indices and one-hot encoded labels.
* wav2mfcc(file_path, n_mfcc=20, max_len=11)
Input: File path to an audio file, number of MFCC features (default is 20), and maximum length of the MFCC features (default is 11).
Output: MFCC features of the audio file.
Description: Loads an audio file, downsamples it, converts it to MFCC features, and pads or truncates the features to the specified length.
* save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20)
Input: Directory path containing labeled subdirectories, maximum length of the MFCC features (default is 11), and number of MFCC features (default is 20).
Output: None
Description: Converts all audio files in each label's directory to MFCC features and saves them as numpy arrays with filenames corresponding to the labels.
* get_train_test(split_ratio=0.8, random_state=42)
Input: Split ratio for train-test split (default is 0.8) and random state for reproducibility (default is 42).
Output: Train and test sets for features (X) and labels (y).
Description: Loads the saved numpy arrays for each label, combines them into a single dataset, and splits the dataset into training and testing sets based on the specified ratio.
* prepare_dataset(path=DATA_PATH)
Input: Directory path containing labeled subdirectories.
Output: Dictionary with labels as keys and corresponding MFCC features as values.
Description: Loads audio files from each label's directory, converts them to MFCC features, and stores them in a dictionary.
* load_dataset(path=DATA_PATH)
Input: Directory path containing labeled subdirectories.
Output: List of tuples containing labels and corresponding MFCC features.
Description: Calls prepare_dataset to get MFCC features for each label and returns the first 100 samples as a dataset.
* Spectograme(path)
Input: File path to an audio file.
Output: None
Description: Loads an audio file, computes its spectrogram, and displays it using matplotlib



2. speech_2_convolutions:
Audio Classification Using Convolutional Neural Networks (CNN)
This project aims to classify audio files into different categories using a Convolutional Neural Network (CNN). The process involves preprocessing the audio data, augmenting it with background noise, and training a CNN model.

Prerequisites
Before running the script, make sure to install the required Python libraries, including TensorFlow, Keras, Librosa, scikit-learn, numpy, tqdm, matplotlib, and scipy.

Directory Structure
Organize your audio files in a directory where each subdirectory represents a label (or class). For example, the main directory could be new_train/, with subdirectories such as bed/, bird/, cat/, etc. Additionally, have a directory for background noise files named _background_noise_/.

Workflow
Data Preprocessing
Save MFCC Features to Array: The script first saves the Mel-frequency cepstral coefficients (MFCC) features of the audio files to numpy arrays. This is done for efficient loading and processing later on.

Load Train and Test Sets: The script then loads the training and testing datasets from the saved numpy arrays.

Reshape the Data: The audio data is reshaped to fit the input requirements of the CNN model.

Data Augmentation
To improve the robustness of the model, background noise is added to the training data. This helps the model generalize better to different audio environments.

Visualize the Data
Optional code is provided to visualize an example of the augmented training data. This helps in understanding how the data looks after preprocessing and augmentation.

One-Hot Encoding
The labels (or classes) are converted to one-hot encoded vectors. This is necessary for training the neural network.

Building the CNN Model
An improved CNN model is built with three convolutional layers, batch normalization, max pooling, dropout layers, and dense layers. This architecture helps in extracting relevant features from the audio data and reduces overfitting.

Compiling and Training the Model
The model is compiled using categorical crossentropy as the loss function and RMSprop as the optimizer. The model is then trained using the training data, with TensorBoard callbacks for visualization.

Evaluating the Model
After training, the model is evaluated on the test data to check its accuracy.

Saving the Model
Finally, the trained model is saved for future use. This allows for easy deployment and reuse without the need to retrain.

