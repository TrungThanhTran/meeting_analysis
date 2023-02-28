import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from functions import *
import validators
from PIL import Image

import textwrap
# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Capturia",page_icon=image_logo)

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
print('state = ', st.session_state)

st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Transcription")
st.markdown("## Transcription Audio")


if "title" not in st.session_state:
    st.session_state.title = ''   
if 'passages' not in st.session_state:
    st.session_state['passages'] = ""
    
passages = st.session_state['passages']
title = st.session_state['title']
            
# sentiment, sentences = sentiment_pipe(passages)
# with st.expander("See Transcribed Text"):
st.write(f"Number of characters: {len(passages)}")
st.subheader(f'Title: {title}')
st.write(passages)
        
        
