from django.shortcuts import render
from django.http import HttpResponse
from django.utils import timezone
import tensorflow as tf
import nltk
import string
import json
import numpy as np
import random
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#from keras.models import Model
from nltk.stem.snowball import SnowballStemmer

snowball = SnowballStemmer("german")
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
model = load_model("chatbot")



with open("qa.json", encoding="utf8") as file:
    data = json.load(file)

labels = []
questions = []
# Tokenize and read Words, Labels, Docs
for intent in data["qa"]:
    for pattern in intent["questions"]:
        questions.append(pattern)
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

training_x = []
for i, question in enumerate(questions):
    # tokenize
    tokens = nltk.word_tokenize(question, language='german')
    # remove punctution
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove non-alphabetic/non-numeric
    real_tokens = [word for word in stripped if word.isalpha() or word.isnumeric()]
    # stemming
    sequence = [snowball.stem(token) for token in real_tokens]

    training_x.append(sequence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_x)

def index(request):
    msg = ""
    if request.method == 'POST':
        user_message = request.POST.get('nachricht', False)
        print(user_message)
        with open("chat.txt", "a+") as f:
            f.write("("+str(timezone.now())+")"+" Sie: ")
            f.write(user_message + "\n")
            f.write("(" + str(timezone.now()) + ")" + " Chatbot: ")
            msg = answer(user_message)
            f.write(msg+"\n")

    f = open("chat.txt", "r")
    chat = f.read()
    return render(request, 'chatting/index.html', {'Chat':  msg})

def response(request):
    msg = ""
    if request.method == 'POST':
        print("hallo")
        user_message = request.POST.get('nachricht', False) #request.POST.get("nachricht")
        print(user_message)
        with open("chat.txt", "a+") as f:
            f.write("("+str(timezone.now())+")"+" Sie: ")
            f.write(user_message + "\n")
            f.write("(" + str(timezone.now()) + ")" + " Chatbot: ")
            msg = answer(user_message)
            f.write(msg+"\n")

    f = open("chat.txt", "r")
    chat = f.read()
    #return render(request, 'chatting/index.html', {'Chat':  msg})
    return HttpResponse(msg)

def answer(user_message):
    preprocessed = []

    # Preprocessing
    # tokenize
    tokens = nltk.word_tokenize(user_message, language='german')
    # remove punctution
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove non-alphabetic/non-numeric
    real_tokens = [word for word in stripped if word.isalpha() or word.isnumeric()]
    # stemming
    sequence = [snowball.stem(token) for token in real_tokens]
    preprocessed.append(sequence)

    print(preprocessed)
    question = tokenizer.texts_to_sequences(preprocessed)
    padded_samples = pad_sequences(question, maxlen=300)

    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        results = model.predict(padded_samples)

    result_index = np.argmax(results)
    tag = labels[result_index]

    for t in data["qa"]:
        if t["tag"] == tag:
            responses = t["answers"]
    print(results)
    print(tag)
    message = random.choice(responses)

    if (results[0][result_index] < 0.5):
        message = "Das habe ich leider nicht so ganz verstanden. :("
    return message