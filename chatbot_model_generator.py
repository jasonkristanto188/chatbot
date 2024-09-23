import os
import json
import nltk
import pickle
import random
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

def modelGenerator(intents_file, model_path, classes_file, words_file):
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()

    intents = json.loads(open(intents_file).read())

    words = []
    classes = []
    documents = []
    innerwords = []
    innerclasses = []
    innerdocuments = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    with open(words_file, 'w') as f:
        json.dump(words, f)

    classes = sorted(set(classes))
    with open(classes_file, 'w') as f:
        json.dump(classes, f)



    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])



    # machine learning stuffs :v
    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save(model_path, hist)
    print('Done')

def update_main_model():
    mainpath = r'D:\Users\jason.kristanto\PycharmProjects\CondaProjects\chatbot\model'
    intents_file = rf'{mainpath}\intents.json'
    model_path = rf'{mainpath}\chatbotmodel.h5'
    classes_file = rf'{mainpath}\classes.json'
    words_file = rf'{mainpath}\words.json'

    modelGenerator(intents_file, model_path, classes_file, words_file)

def update_function_model(project_name):
    # mainpath = rf'D:\Users\jason.kristanto\PycharmProjects\CondaProjects\chatbot\functions\{project_name}\{project_name}_model'
    
    # #khusus panduan_treasury
    mainpath = rf'D:\Users\jason.kristanto\PycharmProjects\CondaProjects\chatbot\functions\panduan_treasury\{project_name}\{project_name}_model'

    if not os.path.isdir(mainpath):
        os.makedirs(mainpath)

    words_file = rf'{mainpath}\words.json'
    classes_file = rf'{mainpath}\classes.json'
    intents_file = rf'{mainpath}\{project_name}_intents.json'
    model_path = rf'{mainpath}\{project_name}_model.h5'

    modelGenerator(intents_file, model_path, classes_file, words_file)

# update_main_model()

project_name = 'Bon Sementara'
update_function_model(project_name)
