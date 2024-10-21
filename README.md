# 🎤 Speech Emotion Recognition System 🎧

Welcome to the **Speech Emotion Recognition** project! This project leverages deep learning techniques to classify emotions from audio data. Using libraries like **TensorFlow**, **Keras**, and **Librosa**, we built a model that can analyze speech clips and predict the emotion behind them.

## 🚀 Project Overview

This project aims to recognize emotions from speech using a **1D Convolutional Neural Network (CNN)**. The model is trained on popular speech emotion datasets like **RAVDESS**, **CREMA**, **TESS**, and **SAVEE**, with over **7,000 audio clips**. The final model achieved **62% accuracy** across 368 emotion categories.

## 📚 Datasets Used
- **RAVDESS** 🎶: Ryerson Audio-Visual Database of Emotional Speech and Song : https://www.kaggle.com/datasets/ejlok1/cremad
- **CREMA-D** 🗣️: Crowd-Sourced Emotional Multimodal Actors Dataset : https://www.kaggle.com/datasets/ejlok1/cremad
- **TESS** 🎼: Toronto Emotional Speech Set : https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- **SAVEE** 🔊: Surrey Audio-Visual Expressed Emotion : https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

## 🛠️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech-emotion-recognition.git
   cd speech-emotion-recognition
2. Install the required dependencies:
   pip install -r requirements.txt
3. Download the datasets and place them in the appropriate directories
4. Run the preprocessing script to extract audio features:
    python preprocess.py
   
# 📊 Model Architecture
The model is a 1D CNN designed for extracting features from raw audio signals. Here's a brief overview of the architecture:

Input Layer: MFCC (Mel-frequency cepstral coefficients) features extracted from the audio files.
Conv1D Layers: To capture temporal features from the speech data.
MaxPooling: To downsample and reduce the dimensionality.
Dropout & Batch Normalization: For regularization and faster convergence.
Dense Layers: Fully connected layers for classification.
Output Layer: Softmax activation to classify among the emotion categories.

# 🔄 Data Preprocessing
We use Librosa to extract MFCC features from the audio files. MFCC is crucial for capturing speech characteristics, and it serves as input to the model. The audio data is resampled to 16kHz, normalized, and converted into spectrograms before feeding it into the CNN.

# 🎯 Results
Accuracy: 62% across 368 emotion categories.
Datasets: Processed over 7,000 audio clips from RAVDESS, CREMA, TESS, and SAVEE.
Prediction Examples: Given an audio clip, the model predicts one of the following emotions: Happy, Sad, Angry, Neutral, etc.

# 🎥 Demo
You can run the model on your own audio files:
python predict.py --audio_file path_to_audio.wav

# 🧑‍💻 How to Use
1. Train the Model: Use the following command to train the model on the preprocessed data.
    python train.py
2. Evaluate the Model: To test the model on the validation set, use:
    python evaluate.py
3. Make Predictions: You can pass any .wav file to the model to predict the emotion:
    python predict.py --audio path_to_file.wav
   
#📝 Conclusion
This project demonstrates the use of deep learning for speech emotion recognition. With further fine-tuning and the use of larger datasets, the model could be extended for real-time emotion recognition in conversational AI systems, call centers, or mental health applications. 🎉

📦 Future Improvements
🔧 Model Optimization: Explore other architectures like RNN or attention-based models to boost performance.
🔍 Dataset Expansion: Incorporate additional datasets for better generalization across languages and accents.
🖥️ Real-time Emotion Detection: Deploy the model in a real-time system for live emotion recognition from speech.
🤝 Contributing
Contributions are welcome! Please submit a pull request or open an issue if you find any bugs or improvements.
