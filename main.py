import streamlit as st
!pip install SpeechRecognition
import speech_recognition as sr
import os
import soundfile as sf

st.write("""
# Halo
Apa *iya*
""")

uploded_file = st.file_uploader("Choose a file")
if uploded_file is not None:
    data, samplerate = sf.read(uploded_file)
    st.write(samplerate)