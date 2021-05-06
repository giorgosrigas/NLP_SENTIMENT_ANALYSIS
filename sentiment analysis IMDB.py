#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import required libraries
import pandas as pd
import numpy as np
import nltk
import sklearn
import operator
import random
from nltk.util import ngrams
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# start importing training, validation and test set
file_neg = pd.read_csv('IMDb/train/imdb_train_neg.txt')
file_pos = pd.read_csv('IMDb/train/imdb_train_pos.txt')
file_neg = np.ravel(dataset_file_neg)
file_pos = np.ravel(dataset_file_pos)

train = []
for pos_review in file_pos:
    train.append((pos_review, 1))
for neg_review in file_neg:
    train.append((neg_review, 0))

file_neg = pd.read_csv('IMDb/dev/imdb_dev_neg.txt')
file_pos = pd.read_csv('IMDb/dev/imdb_dev_pos.txt')
file_neg = np.ravel(file_neg)
file_pos = np.ravel(file_pos)

val = []
for pos_review in file_pos:
    val.append((pos_review, 1))
for neg_review in file_neg:
    val.append((neg_review, 0))

file_neg = pd.read_csv('IMDb/test/imdb_test_neg.txt')
file_pos = pd.read_csv('IMDb/test/imdb_test_pos.txt')
file_neg = np.ravel(file_neg)
file_pos = np.ravel(file_pos)

test = []
for pos_review in file_pos:
    test.append((pos_review, 1))
for neg_review in file_neg:
    test.append((neg_review, 0))

print()
print("Training size: " + str(len(train)))
print("Validation size: " + str(len(dev)))
print("Test size: " + str(len(test)))

# call stop words from nltk
lemmatizer = nltk.stem.WordNetLemmatizer()
stop = set(nltk.corpus.stopwords.words('english'))
stop.add(".")
stop.add(",")
stop.add("``")
stop.add("#")
stop.add(":")
stop.add("/")
stop.add(">")
stop.add("<")
stop.add("br")
stop.add("(")
stop.add(")")

def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


def get_vector_text(list_vocab, string):
    vector_text = np.zeros(len(list_vocab) + 1)
    list_tokens_string = get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i] = list_tokens_string.count(word)
    #   Putting as last feature the word count
    vector_text[len(list_vocab)] = len(string.split())
    return vector_text


def get_vocabulary(training_set, num_features):  # Function to retrieve vocabulary
    dict_word_frequency = {}
    for instance in training_set:
        sentence_tokens = get_list_tokens(instance[0])
     
        # find bigrams that express sentiment
        n_grams = ngrams(sentence_tokens, 2)
        n_gram_list = [' '.join(grams) for grams in n_grams]
        for word in n_gram_list:
            if word in stop: continue
            if any(stopword in stop for stopword in word.split()): continue
            if word not in dict_word_frequency:
                dict_word_frequency[word] = 1
            else:
                dict_word_frequency[word] += 1
        
        # find single words that express sentiment
        for word in sentence_tokens:
            if word in stop: continue
            if word not in dict_word_frequency:
                dict_word_frequency[word] = 1
            else:
                dict_word_frequency[word] += 1
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
    vocabulary = []
    for word, frequency in sorted_list:
        vocabulary.append(word)
    return vocabulary

# function to train the model
def train_classifier(training_set, vocabulary):
    X_train = []
    Y_train = []
    for instance in training_set:
        vector_instance = get_vector_text(vocabulary, instance[0])
        X_train.append(vector_instance)
        Y_train.append(instance[1])
    # svm classifier
    svm =  SVC(kernel = 'linear')
    svm.fit(np.asarray(X_train), np.asarray(Y_train))
    return svm


Y_dev = []
for instance in dev:
    Y_dev.append(instance[1])
Y_dev_gold = np.asarray(Y_dev)


# list with possible number of feature that could optimize the model in the validation set
list_num_features = [100, 200, 500, 1000, 1250, 1750]
best_accuracy_dev = 0
for num_features in list_num_features:
    # First, we extract the vocabulary from the training set and train our model
    vocabulary = get_vocabulary(train, num_features)
    svm = train_classifier(train, vocabulary)
    
    # we transform our dev set into vectors and make the prediction on this set
    X_dev = []
    for instance in dev:
        vector_instance = get_vector_text(vocabulary, instance[0])
        X_dev.append(vector_instance)
    X_dev = np.asarray(X_dev)
    Y_dev_predictions = svm.predict(X_dev)
    # Getting and printing the results of the classifier
    accuracy_dev = accuracy_score(Y_dev_gold, Y_dev_predictions)
    precision = precision_score(Y_dev_gold, Y_dev_predictions)
    recall = recall_score(Y_dev_gold, Y_dev_predictions)
    f1 = f1_score(Y_dev_gold, Y_dev_predictions)

    print("       " + str(num_features) + "           " + str(round(precision, 3)) +
          "         " + str(round(recall, 3)) + "         " + str(round(f1, 3)) + "         " +
          str(round(accuracy_dev, 3)))
    
    # Seeking for the best accuracy
    if accuracy_dev >= best_accuracy_dev:
        best_accuracy_dev = accuracy_dev
        best_num_features = num_features
        best_vocabulary = vocabulary
        best_log_reg = log_reg
print("Best accuracy overall in the dev set is " + str(round(best_accuracy_dev, 3)) + " with " +
      str(best_num_features) + " features.")


# Creating a vocabulary for the best number of features found above
vocabulary = get_vocabulary(train, 1000)
svm = train_classifier(train, vocabulary)


# Make predictions on test test
X_test = []
Y_test = []
for instance in test:
    vector_instance = get_vector_text(vocabulary, instance[0])
    X_test.append(vector_instance)
    Y_test.append(instance[1])
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
Y_test_predictions = svm.predict(X_test)

precision = precision_score(Y_test, Y_test_predictions)
recall = recall_score(Y_test, Y_test_predictions)
f1 = f1_score(Y_test, Y_test_predictions)
accuracy = accuracy_score(Y_test, Y_test_predictions)

print("Test Precision: " + str(round(precision, 3)))
print("Test Recall: " + str(round(recall, 3)))
print("Test F1-Score: " + str(round(f1, 3)))
print("Test Accuracy: " + str(round(accuracy, 3)))

