import numpy as np
import keras
import keras.datasets.imdb as imdb
from keras import layers
from keras.models import Sequential
from keras.preprocessing import sequence
from gensim.models import Word2Vec
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import keras_nlp
from tensorflow_hub import KerasLayer

'''positive = 'some_positive_value'
negative = 'some_negative_value'

additional_kwargs = {
    'key1': positive,
    'key2': negative,
    # Add more key-value pairs as needed
}'''

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)

# Create a reverse word index
max_features = 20000  # Only consider the top 30k words
maxlen = 200  # Only consider the first 300 words of each movie review
vector_size = 32 # Size of Word2Vec word vector
word_index = imdb.get_word_index() # word indexes, ranked the lower index,appears more often 
reverse_word_index = {value: key for key, value in word_index.items()} # for translation of indexes back to words
# Numpy arrays of cleaned, preprocessed ,indexed reviews, which we will use to
(x_train, y_train), (x_val, y_val) = imdb.load_data(
    num_words=max_features
)
#sample row, sample index
print(x_train[0][:15]," ".join([reverse_word_index[x-3]if x>3 else "" for x in x_train[0][:15]]))
def encode_sentence(sentence, max_words=max_features):
    words = sentence.split()
    words = [word_index[word] + 3 for word in words if word in word_index and word_index[word] + 3 < max_words]
    return words

# Function to decode integers back to words
def decode_integers(integers):
    reverse_word_index[0] = "<PAD>"  # Padding
    reverse_word_index[1] = "<START>"  # Start of sequence
    reverse_word_index[2] = "<UNKNOWN>"  # Unknown word
    reverse_word_index[3] = "<UNUSED>"  # Unused
    words = [reverse_word_index.get(i-3, "<UNKNOWN>") for i in integers]
    return words

# Function to map words to Word2Vec vectors
def map_words_to_vectors(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.array(vectors)
print(decode_integers(x_train[0][:15]))
sentences = [decode_integers(x) for x in x_train]  # Assuming x_train is a list of sentences

w2v_model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)

x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)
# Word2Vec dataset for LSTM
X_w2v_train = []
X_w2v_val = []
# Text dataset for BERT
x_train_sentences = []
x_val_sentences = []

# Loop through each sequence in x_train
for sequence in x_train:
    decoded_review = decode_integers(sequence)
    word_vectors = map_words_to_vectors(decoded_review, w2v_model)
    X_w2v_train.append(word_vectors)
    x_train_sentences.append(" ".join(decoded_review))

for sequence in x_val:
    decoded_review = decode_integers(sequence)
    word_vectors = map_words_to_vectors(decoded_review, w2v_model)
    X_w2v_val.append(word_vectors)
    x_val_sentences.append(" ".join(decoded_review))
    
# Convert the list to a numpy array
X_w2v_train = np.array(X_w2v_train)
X_w2v_val = np.array(X_w2v_val)
x_train_sentences=np.array(x_train_sentences)
x_val_sentences=np.array(x_val_sentences)
model_1 = Sequential([
    layers.Input(shape=(maxlen, vector_size),),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64,),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")])
model_1.summary()

model_1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history_1=model_1.fit(X_w2v_train, y_train, batch_size=128, epochs=5, validation_data=(X_w2v_val, y_val))

y_pred_1=model_1.predict(X_w2v_val)
pd.DataFrame(history_1.history).plot(figsize=(8,5))
plt.show()
# Pretrained classifier.
model_2 = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
)
model_2.backbone.trainable = False
model_2.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(1e-4),
    jit_compile=True,
)
model_2.summary()
#model.to(device)
#history_2=model_2.fit(x=x_train_sentences, y=y_train, epochs=3, batch_size=128)
# Test Predictions
y_pred_2 = model_2.predict(x_val_sentences)
