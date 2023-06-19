import streamlit as st
import os
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
import pocketsphinx
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

#Training
data = pd.read_csv('twitter_training.csv')
#Cleaning
data = pd.DataFrame.drop(data, columns=['id', 'entity'])
data = pd.DataFrame.dropna(data, axis=0)
#BagOfWord
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer().fit(data['text'])
bag_of_words = vectorizer.transform(data['text'])
#Training Naive Baiytes
x_train = bag_of_words
y_train = data['sentiment']
nb = MultinomialNB()
nb.fit(x_train, y_train)
#Training SGD Classifier
clf = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(loss = 'log_loss' , max_iter=1000, tol=1e-3))
clf.fit(x_train, y_train)

def pred(text):
    #Predict Naive Bayes
    test_data = pd.DataFrame([text], columns=['text'])
    x_test1 = vectorizer.transform(test_data['text'])
    prediction = nb.predict(x_test1)
    hasil = prediction[0]
    st.write("**Naive Bayes :** ", hasil)
    #Predict Logistic Regression
    prediction = clf.predict(x_test1)
    hasil = prediction[0]
    st.write("**SGD :** ", hasil)

#Load UI
st.write("""
# Speech Recognition
""")

source = st.selectbox(
        'Choose source',
        ('Choose', 'Text', 'Audio File'))
if source == "Text":
    input=""
    text = st.text_input('Input Text')
    predict = st.button("Predict")
    if predict:
        pred(text)
elif source == "Audio File":
    uploded_file = st.file_uploader("Choose a file")
    if uploded_file is not None:
        sr.AudioFile(uploded_file)
        r = sr.Recognizer()
        input = ""
        with sr.AudioFile(uploded_file.name) as source:
            AudioData = r.record(source)
            option = st.selectbox(
            'Choose method',
            ('Choose', 'Google', 'Sphinx'))
            if option == "Google":
                googleResult = r.recognize_google(AudioData)
                label = "Result with Google Recognizer"
                input = googleResult
            elif option == "Sphinx":
                sphinxResult = r.recognize_sphinx(AudioData)
                label = "Result with Sphinx Recognizer"
                input = sphinxResult
            text = st.text_input(label, input)
            predict = st.button("Predict")
            if predict:
                pred(text)