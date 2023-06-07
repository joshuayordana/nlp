import streamlit as st
import os
from pydub import AudioSegment, silence

st.write("""
# Halo
Apa *iya*
""")

uploded_file = st.file_uploader("Choose a file")
if uploded_file is not None:
    data, samplerate = sf.read(uploded_file)
    st.write(samplerate)