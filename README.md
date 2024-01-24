IMDb Sentiment Analysis Project
This project focuses on sentiment analysis of IMDb movie reviews using Word2Vec embeddings and LSTM (Long Short-Term Memory) neural networks, along with fine-tuning a pre-trained DistilBERT model for comparison. Sentiment analysis helps determine the sentiment (positive or negative) expressed in a movie review.

Table of Contents
Overview
Dependencies
Usage
Approaches
Results
Contributing
License
Overview
This repository contains a Python script for sentiment analysis on the IMDb movie reviews dataset. The sentiment analysis is performed using two different approaches:

Word2Vec and LSTM:

Word2Vec embeddings represent words in movie reviews.
LSTM neural network is employed for sentiment classification.
Word sequences are preprocessed and padded to a fixed length before training the LSTM model.
BERT (Bidirectional Encoder Representations from Transformers):

A pre-trained DistilBERT model is employed for sentiment classification.
The DistilBERT model is fine-tuned on the IMDb dataset for improved performance.
Dependencies
Ensure you have the following dependencies installed:

Python
NumPy
Keras
Gensim
Scikit-learn
Matplotlib
TensorFlow
Keras-NLP
TensorFlow Hub
You can install the required dependencies using:

bash
Copy code
pip install -r requirements.txt
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/imdb-sentiment-analysis.git
Navigate to the project directory:

bash
Copy code
cd imdb-sentiment-analysis
Run the script:

bash
Copy code
python sentiment_analysis.py
Approaches
Word2Vec and LSTM:

Word sequences are converted to Word2Vec embeddings.
LSTM model is trained for sentiment analysis.
Results are visualized using Matplotlib.
BERT (DistilBERT):

A pre-trained DistilBERT model is used.
The model is fine-tuned on IMDb dataset for sentiment analysis.
Model summary and results are displayed.
Results
LSTM Model:

Training accuracy, validation accuracy, and loss are visualized using Matplotlib.
BERT Model:

Model summary and performance metrics are displayed.
Feel free to experiment with different configurations, hyperparameters, and model architectures to enhance sentiment analysis performance.

Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements.

License
This project is licensed under the MIT License.
