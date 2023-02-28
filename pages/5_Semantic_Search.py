import streamlit as st
from functions import *
from PIL import Image

# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Capturia",page_icon=image_logo)

# st.set_page_config(page_title="Question/Answering", page_icon="🔎")
st.sidebar.header("Semantic Search")
st.markdown("##  Semantic Search")

def gen_sentiment(text):
    '''Generate sentiment of given text'''
    return sent_pipe(text)[0]['label']

def gen_annotated_text(df):
    '''Generate annotated text'''
    
    tag_list=[]
    for row in df.itertuples():
        label = row[2]
        text = row[1]
        if label == 'Positive':
            tag_list.append((text,label,'#8fce00'))
        elif label == 'Negative':
            tag_list.append((text,label,'#f44336'))
        else:
            tag_list.append((text,label,'#000000'))
        
    return tag_list

bi_enc_dict = {'mpnet-base-v2':"all-mpnet-base-v2",
              'instructor-base': 'hkunlp/instructor-base',
              'setfit-finance': 'nickmuchi/setfit-finetuned-financial-text-classification'}

search_input = st.text_input(
        label='Enter Your Search Query',value= "ex: what kind of bussiness?", key='search')
        
sbert_model_name = st.sidebar.selectbox("Embedding Model", options=list(bi_enc_dict.keys()), key='sbox')
        
chunk_size = st.sidebar.slider("Number of Words per Chunk of Text",min_value=100,max_value=250,value=200)
overlap_size = st.sidebar.slider("Number of Overlap Words in Search Response",min_value=30,max_value=100,value=50)
chain_type = st.sidebar.radio("Langchain Chain Type",options = ['Normal','Refined'])

try:

    if search_input:
        
        if "sen_df" in st.session_state and "passages" in st.session_state:
        
            ## Save to a dataframe for ease of visualization
            sen_df = st.session_state['sen_df']

            title = st.session_state['title']

            embedding_model = bi_enc_dict[sbert_model_name]
                            
            with st.spinner(
                text=f"Loading {embedding_model} embedding model and Generating Response..."
            ):
                
                docsearch = process_corpus(st.session_state['passages'],emb_tokenizer,title, embedding_model)

                result = embed_text(search_input,title,embedding_model,emb_tokenizer,docsearch,chain_type)
                

            references = [doc.page_content for doc in result['input_documents']]

            answer = result['output_text']

            sentiment_label = gen_sentiment(answer)
                
            ##### Sematic Search #####
            
            df = pd.DataFrame.from_dict({'Text':[answer],'Sentiment':[sentiment_label]})
              
            
            text_annotations = gen_annotated_text(df)[0]            
            
            with st.expander(label='Query Result', expanded=True):
                annotated_text(text_annotations)
                
            with st.expander(label='References from Corpus used to Generate Result'):
                for ref in references:
                    st.write(ref)
                
        else:
            
            st.write('Please ensure you have entered the YouTube URL or uploaded audio file')
            
    else:
    
        st.write('Please ensure you have entered the YouTube URL or uploaded audio file')  
        
except Exception as e:
    st.write("This function required API call to OPEN AI, please buy credits from them")
    # st.write('Please ensure you have entered the YouTube URL or uploaded audio file')        

    
