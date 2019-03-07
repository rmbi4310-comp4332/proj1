import numpy as np
import string
import pandas as pd
import nltk
import keras

from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adam
from keras import metrics

stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
def prep_data(fname, input_length, top_words, tokenizer=None):
    '''
    :param fname: path for the csv dataset
    :param input_length: the length of sequences, type: int
    :param top_words: the num_words to use as tokenizer corpus
    :param tokenizer: initialized tokenizer with train_df['text'], type: keras.preprocessing.text.Tokenizer
    '''
    df = pd.read_csv(fname)

    text = df['text'].values
    if tokenizer == None:
        tokenizer = Tokenizer(num_words=top_words)
        tkn_corpus = text
        tokenizer.fit_on_texts(tkn_corpus)

    data_matrix = tokenizer.texts_to_sequences(text)

    # vocab_size = len(tokenizer.word_index) + 1

    padded_data_matrix = pad_sequences(data_matrix, padding='post', maxlen=input_length)

    stars = df['stars'].apply(int) -1

    return df['review_id'], stars, padded_data_matrix, tokenizer
# ----------------- End of Helper Functions-----------------

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]


    np.save('embedding_matrix', embedding_matrix)


    return embedding_matrix

def load_data(input_length, top_words=5000):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, t_tokenizer = prep_data("data/train.csv", input_length, top_words)
    K = max(train_data_label)+1  # labels begin with 0

    # Load valid data
    valid_id_list, valid_data_label, valid_data_matrix, _ = prep_data("data/train.csv", input_length, top_words, tokenizer=t_tokenizer)

    # Load testing data
    test_id_list, _, test_data_matrix, _ = prep_data("data/test.csv", input_length, top_words, tokenizer=t_tokenizer)
    
    print("Vocabulary Size:", len(t_tokenizer.word_index)+1)
    print("Training Set Size:", len(train_id_list))
    print("Validation Set Size:", len(valid_id_list))
    print("Test Set Size:", len(test_id_list))
    print("Training Set Shape:", train_data_matrix.shape)
    print("Validation Set Shape:", valid_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    valid_data_label = keras.utils.to_categorical(valid_data_label, num_classes=K)
    return train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, None, t_tokenizer


if __name__ == '__main__':
    # Hyperparameters
    input_length = 300
    embedding_size = 50
    hidden_size = 128
    batch_size = 100
    dropout_rate = 0.5
    learning_rate = 0.2
    total_epoch = 5

    train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, _, t_tokenizer = load_data(input_length, 5000)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(t_tokenizer.word_index) + 1
    output_size = K

    # New model
    model = Sequential()
    embedding_dim = embedding_size
    # embedding_matrix = create_embedding_matrix(
    #     'glove.6B/glove.6B.100d.txt',
    #     t_tokenizer.word_index, embedding_dim)
    # OR

    # load existing embedding_matrix  
    embedding_matrix = np.load('embedding_matrix.npy')
    
    # embedding layer and dropout
    # YOUR CODE HERE
    model.add(Embedding(input_dim=input_size, output_dim=embedding_size,weights=[embedding_matrix],
                         input_length=input_length, trainable = False))
    model.add(Dropout(dropout_rate))

    
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM layer
    # YOUR CODE HERE
    # model.add(LSTM(units=hidden_size, dropout=dropout_rate))
    model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
    

    # output layer
    # YOUR CODE HERE
    model.add(Dropout(0.5))
    model.add(Dense(K, activation='softmax'))

    # SGD optimizer with momentum
    # optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=learning_rate)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # training
    model.fit(train_data_matrix, train_data_label, epochs=total_epoch, batch_size=batch_size)
    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))
    valid_score = model.evaluate(valid_data_matrix, valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    test_pre = model.predict(test_data_matrix, batch_size=batch_size).argmax(axis=-1) + 1
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_id_list
    sub_df["pre"] = test_pre
    sub_df.to_csv("pre.csv", index=False)
