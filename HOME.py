import whisper
import os
from pytube import YouTube
import pandas as pd
import plotly_express as px
import nltk
import plotly.graph_objects as go
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import streamlit as st
import en_core_web_lg
from PIL import Image
from functions import *
from st_custom_components import st_audiorec
# import streamlit.components.v1 as components
# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Capturia",page_icon=image_logo)

nltk.download('punkt')

from nltk import sent_tokenize

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Home")
asr_model_options = ['small.en', 'base.en','tiny.en']
asr_model_name = st.sidebar.selectbox("Whisper Model Options", options=asr_model_options, key='sbox')

col1, col2 = st.columns(2)
with col1:
    original_title = '<center><p style="font-size: 80px;">Capturia</p> \n <p>AI MEETING NOTES & SENTIMENT ANALYSIS </p></center>'
    st.markdown(original_title, unsafe_allow_html=True)
    # st.title("Capturia " + "\n"
    #              "AI MEETING NOTES & SENTIMENT ANALYSIS ",100)
with col2:
    image = Image.open('data/logo/logo.png')
    st.image(image,width=200)

st.markdown("<br><br><br>",  unsafe_allow_html=True)
print('state = ', st.session_state)

### prepare state for 
if 'sbox' not in st.session_state:
    st.session_state.sbox = asr_model_name

if "sen_df" not in st.session_state:
    st.session_state['sen_df'] = ''

def clean_directory(paths):
    for path in paths:
        if not os.path.exists(path):
            pass
        else:
            for file in os.listdir(path):
                if ("mp3" in file) or ("mp4" in file):
                    os.remove(os.path.join(path, file))

### Preload model
try:
    ASR_MODEL = load_asr_model(st.session_state.sbox)
except Exception as e:
    print(e)
    st.session_state.sbox = 'small.en'
    ASR_MODEL = load_asr_model('small.en')
    
def infer_audio(state, asr_model, type):
    results, title = inference(state, asr_model, type)
    print('results = ', results)
    # passages = results
    passages = clean_text(results)
        
    st.session_state['passages'] = passages
    st.session_state['title'] = title

st.markdown("## Please submit your audio or video file",  unsafe_allow_html=True)

### UPLOAD AND PROCESS
choice = st.radio("", ["By uploading a file","By getting from Youtube URL"]) 
if choice:
    clean_directory(["./temp", "./temp/youtube"])
    if choice == "By uploading a file":
        upload_wav = st.file_uploader("Upload a .wav sound file ",key="upload")
        if upload_wav:
            with open(os.path.join("./temp/audio.mp3"),"wb") as f:
                f.write(upload_wav.getbuffer())
            st.session_state['url'] = ''
            
    elif choice == "By getting from Youtube URL":
        url_input = st.text_input(
        label="Enter YouTube URL, below is a calling example", value='https://www.youtube.com/watch?v=agizP0kcPjQ',
        key="url")
        if 'upload' in st.session_state:
            st.session_state['upload'] = ''
        
    btn_transcribe = st.button('Transcribe')
    if btn_transcribe:
        if 'passage' in st.session_state:
            st.session_state['passage'] = ""
            
        with st.spinner(text="Transcribing..."):
            try:
                if ('url' in st.session_state) and (st.session_state['url'] != ''):
                    if len(st.session_state['url']) > 0: 
                        infer_audio(st.session_state['url'], ASR_MODEL, type='url') 

                if ('upload' in st.session_state) and (st.session_state['upload'] != ''):  
                    if st.session_state['upload'] is not None:
                        infer_audio(st.session_state['upload'], ASR_MODEL, type='upload')
                
            except Exception as e:
                print(e)
                st.write("No YouTube URL or file upload detected")
        st.success('Transcribing audio done!')
    
# if "url" not in st.session_state:
#     st.session_state.url = "https://www.youtube.com/watch?v=agizP0kcPjQ"
    
# st.markdown(
#     "<h3 style='text-align: center; color: red;'>OR</h3>",
#     unsafe_allow_html=True
# )

auth_token = os.environ.get("auth_token")

