# RNN-Movie-Review-Analyzer

Project Overview
This project implements a Deep Learning solution using a Bidirectional Long Short-Term Memory (Bi-LSTM) network to classify movie reviews from the IMDb dataset as either Positive or Negative. This demonstrates proficiency in Natural Language Processing (NLP), recurrent neural network architectures, and full end-to-end deployment using Streamlit.

🚀 Getting Started
To run the interactive web application demo on your local machine, follow these steps:

Prerequisites
You need Python 3.8+ installed. All required dependencies are listed in the requirements.txt file (which you can see in the Canvas).

1. Setup Environment
Navigate to the project directory in your terminal and install the necessary libraries:

pip install -r requirements.txt

2. Required Assets
Ensure the following pre-trained files are present in the root directory:

bilstm_sentiment_model.h5: The saved, trained Keras model.

word_index.json: The mapping of words to tokens used during training.

app.py: The Streamlit web application script.

3. Run the Demo
Execute the Streamlit script to launch the interactive web application:

streamlit run app.py

The application will open automatically in your browser, allowing you to paste in custom movie reviews for instant sentiment analysis.

🧠 Model Architecture
The final, production-ready model utilizes a Bi-LSTM architecture, which processes the text sequence in both forward and backward directions, greatly improving its ability to handle negation and capture long-range dependencies.

Model Parameters:

max_words (Vocabulary Size): 10,000

maxlen (Padded Sequence Length): 200

embedding_dim: 128

lstm_units: 64 (Bidirectional)

Layer (type)

Output Shape

Parameters

Embedding

(None, 200, 128)

1,280,000

Bidirectional (LSTM)

(None, 128)

82,944

Dropout

(None, 128)

0

Dense

(None, 1)

129

Total Params

1,363,073



📊 Results and Performance
The model was trained on the IMDb Movie Review Dataset (25,000 training samples, 25,000 test samples). Training was halted early using Early Stopping (patience=3) to prevent overfitting.

Metric

Score

Test Accuracy

84.89%

F1-Score (Positive)

0.85

F1-Score (Negative)

0.85

Training Epochs

4 (Stopped at Epoch 4 due to Early Stopping)

Visualizations
Note: The training history plot above confirms that the model showed slight overfitting starting around Epoch 2 (where validation loss began to rise while training loss continued to drop), validating the use of the Early Stopping callback.

Note: The Confusion Matrix shows the number of True Positives, True Negatives, False Positives, and False Negatives, detailing the model's error distribution.
