import streamlit as st
import os
import soundfile as sf
import matplotlib.pyplot as plt
import speech_recognition as sr

st.write("""
# Speech Recognition
Apa *iya*
""")

uploded_file = st.file_uploader("Choose a file")
if uploded_file is not None:
    data, samplerate = sf.read(uploded_file)
    st.write("Data shape:", len(data))
    st.write("Sample rate:", samplerate)
    sr.AudioFile(uploded_file)
    r = sr.Recognizer()
    with sr.AudioFile(uploded_file.name) as source:
        AudioData = r.record(source)
        alternatives = r.recognize_google(AudioData, show_all=True)
        st.write(alternatives)
