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
if "passages" not in st.session_state:
    st.session_state["passages"] = ''
    
if "sen_df" not in st.session_state:
    st.session_state['sen_df'] = ''

def clean_directory(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
clean_directory("./temp")


### Preload model
try:
    ASR_MODEL = load_asr_model(st.session_state.sbox)
except Exception as e:
    print(e)
    st.session_state.sbox = 'small.en'
    ASR_MODEL = load_asr_model('small.en')
    
def infer_audio(state, asr_model, type):
    results, title = inference(state, asr_model, type)
    # passages = results
    passages = clean_text(results)
        
    st.session_state['passages'] = passages
    st.session_state['title'] = title

st.markdown("## Give us audio",  unsafe_allow_html=True)
st.write(st.session_state)

### UPLOAD AND PROCESS
choice = st.radio("", ["By uploading a file","By getting from Youtube URL"]) 
if choice:
    # if choice == "By starting record":
    #     record = audiorecorder("Click to record", "Recording...")
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
        if 'update' in st.session_state:
            st.session_state['update'] = ''
    
    # elif choice == "By recording":
    #     wav_audio_data = st_audiorec()
    #     if wav_audio_data:
    #         st.audio(wav_audio_data, format="audio/wav")
    #         wav_file = open("./temp/record.mp3", "wb")
    #         wav_file.write(wav_audio_data.tobytes())
    #         wav_file.close()
        
    btn_transcribe = st.button('Transcribe')
    if btn_transcribe:
        with st.spinner(text="In progress..."):
            try:
                if "url" in st.session_state:
                    if len(st.session_state['url']) > 0: 
                        infer_audio(st.session_state['url'], ASR_MODEL, type='url') 

                if "upload" in st.session_state:  
                    if st.session_state['upload'] is not None:
                        print('using this function') 
                        infer_audio(st.session_state['upload'], ASR_MODEL, type='upload')
                
                # if "record" in st.session_state:
                #     if st.session_state['record'] is not None:
                #         ./temp/record.mp3
            except Exception as e:
                print(e)
                st.write("No YouTube URL or file upload detected")
        st.success('Processing audio done!')
    
# if "url" not in st.session_state:
#     st.session_state.url = "https://www.youtube.com/watch?v=agizP0kcPjQ"
    
# st.markdown(
#     "<h3 style='text-align: center; color: red;'>OR</h3>",
#     unsafe_allow_html=True
# )

auth_token = os.environ.get("auth_token")

