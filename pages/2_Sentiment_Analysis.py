import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from functions import *
from optimum.onnxruntime import ORTModelForSequenceClassification
import validators
import textwrap
from PIL import Image

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
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Sentiment Analysis using AI")
st.markdown("## Sentiment Analysis")

if "title" not in st.session_state:
    st.session_state.title = ''   
try:
    if ("passages" in st.session_state) and (len(st.session_state['passages']) > 0):
        sentiment, sentences = sentiment_pipe(st.session_state['passages'])
        
        ## Save to a dataframe for ease of visualization
        sen_df = pd.DataFrame(sentiment)
        sen_df['text'] = sentences
        grouped = pd.DataFrame(sen_df['label'].value_counts()).reset_index()
        grouped.columns = ['sentiment','count']
        
        st.session_state['sen_df'] = sen_df
        
        # Display number of positive, negative and neutral sentiments
        fig = px.bar(grouped, x='sentiment', y='count', color='sentiment', color_discrete_map={"Negative":"firebrick","Neutral":\
                                                                                               "navajowhite","Positive":"darkgreen"},\
                                                                                               title='Sentiment Analysis')
        fig.update_layout(
        	showlegend=False,
            autosize=True,
            margin=dict(
                l=25,
                r=25,
                b=25,
                t=50,
                pad=2
            )
        )
        
        st.plotly_chart(fig)
        
        ## Display sentiment score
        try:
            pos_perc = grouped[grouped['sentiment']=='Positive']['count'].iloc[0]*100/sen_df.shape[0]
            neg_perc = grouped[grouped['sentiment']=='Negative']['count'].iloc[0]*100/sen_df.shape[0]
            neu_perc = grouped[grouped['sentiment']=='Neutral']['count'].iloc[0]*100/sen_df.shape[0]
            
            sentiment_score = neu_perc+pos_perc-neg_perc
            
            fig_1 = go.Figure()
            
            fig_1.add_trace(go.Indicator(
                mode = "delta",
                value = sentiment_score,
                domain = {'row': 1, 'column': 1}))
            
            fig_1.update_layout(
                template = {'data' : {'indicator': [{
                    'title': {'text': "Sentiment Score"},
                    'mode' : "number+delta+gauge",
                    'delta' : {'reference': 50}}]
                                    }},
                autosize=False,
                width=250,
                height=250,
                margin=dict(
                    l=5,
                    r=5,
                    b=5,
                    pad=2
                )
            )
        
            with st.sidebar:
            
                st.plotly_chart(fig_1)
            hd = sen_df.text.apply(lambda txt: '<br>'.join(textwrap.wrap(txt, width=70)))
            ## Display negative sentence locations
            fig = px.scatter(sen_df, y='label', color='label', size='score', hover_data=[hd], color_discrete_map={"Negative":"firebrick","Neutral":"navajowhite","Positive":"darkgreen"}, title='Sentiment Score Distribution')
            fig.update_layout(
                showlegend=False,
                autosize=True,
                width=800,
                height=500,
                margin=dict(
                    b=5,
                    t=50,
                    pad=4
                )
            )
        
            st.plotly_chart(fig)
        except Exception as e:
            st.error('The input audio is too short. Cannot provide a Sentiment Analysis!')
        
    else:
        st.write("No YouTube URL or file upload detected")
        
except (AttributeError, TypeError):
    st.write("No YouTube URL or file upload detected")
    
