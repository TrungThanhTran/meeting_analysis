import streamlit as st
from functions import *

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Summarization")
st.markdown("## Summarization using AI")

max_len= st.slider("Maximum length of the summarized text",min_value=70,max_value=200,step=10,value=100)
min_len= st.slider("Minimum length of the summarized text",min_value=20,max_value=200,step=10)

st.markdown("####")     
        
st.subheader("Summarized with matched Entities")

if "passages" not in st.session_state:
    st.session_state["passages"] = ''

if st.session_state['passages']:
      
    with st.spinner("Summarizing and matching entities, this takes a few seconds..."):
        
        try:
            text_to_summarize = chunk_and_preprocess_text(st.session_state['passages'])
            summarized_text = summarize_text(text_to_summarize,max_len=max_len,min_len=min_len)
            
        
        except IndexError:
            try:
                
                text_to_summarize = chunk_and_preprocess_text(st.session_state['passages'],450)
                summarized_text = summarize_text(text_to_summarize,max_len=max_len,min_len=min_len)
                
    
            except IndexError:
                
                text_to_summarize = chunk_and_preprocess_text(st.session_state['passages'],400)
                summarized_text = summarize_text(text_to_summarize,max_len=max_len,min_len=min_len)
                        
        entity_match_html = highlight_entities(text_to_summarize,summarized_text)
        st.markdown("####")
        
        with st.expander(label='Summarized  Audio',expanded=True): 
            st.write(entity_match_html, unsafe_allow_html=True)
        
        st.markdown("####")     
        
        summary_downloader(summarized_text)
            
else:
      st.write("No text to summarize detected, please ensure you have entered the YouTube URL on the Sentiment Analysis page")
      
