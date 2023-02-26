import whisper
import os
from pytube import YouTube
import pandas as pd
import plotly_express as px
import nltk
import plotly.graph_objects as go
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import streamlit as st
import en_core_web_lg
import validators
import re
import itertools
import numpy as np
from bs4 import BeautifulSoup   
import base64, time
from annotated_text import annotated_text
import pickle, math
import wikipedia
from pyvis.network import Network
import torch
from pydub import AudioSegment

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain import VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.prompts.base import RegexParser

nltk.download('punkt')


from nltk import sent_tokenize

OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')
time_str = time.strftime("%d%m%Y-%H%M%S")
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; 
margin-bottom: 2.5rem">{}</div> """

#Stuff Chain Type Prompt template
output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{summaries}
---------
Question: {question}
Helpful Answer:"""

#Refine Chain Type Prompt Template
refine_prompt_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_prompt_template,
)


initial_qa_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n.\n"
)

###################### Functions #######################################################################################

@st.experimental_singleton(suppress_st_warning=True)
def load_models():

    '''Load and cache all the models to be used'''
    q_model = ORTModelForSequenceClassification.from_pretrained("nickmuchi/quantized-optimum-finbert-tone")
    ner_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    kg_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    kg_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    q_tokenizer = AutoTokenizer.from_pretrained("nickmuchi/quantized-optimum-finbert-tone")
    ner_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    emb_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    sent_pipe = pipeline("text-classification",model=q_model, tokenizer=q_tokenizer)
    sum_pipe = pipeline("summarization",model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn",clean_up_tokenization_spaces=True)
    ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=True)
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1') #cross-encoder/ms-marco-MiniLM-L-12-v2
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    return sent_pipe, sum_pipe, ner_pipe, cross_encoder, kg_model, kg_tokenizer, emb_tokenizer, sbert

@st.experimental_singleton(suppress_st_warning=True)
def load_asr_model(asr_model_name):
    asr_model = whisper.load_model(asr_model_name)
    
    return asr_model

@st.experimental_singleton(suppress_st_warning=True)
def process_corpus(corpus, _tokenizer, title, embedding_model, chunk_size=200, overlap=50):

    '''Process text for Semantic Search'''
    
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(_tokenizer,chunk_size=chunk_size,chunk_overlap=overlap,separator='.')

    texts = text_splitter.split_text(corpus)

    embeddings = gen_embeddings(embedding_model)

    docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])

    return docsearch

@st.experimental_singleton(suppress_st_warning=True)
def chunk_and_preprocess_text(text,thresh=500):
    
    """Chunk text longer than n tokens for summarization"""
    
    sentences = sent_tokenize(clean_text(text))
    #sentences = [i.text for i in list(article.sents)]
    
    current_chunk = 0
    chunks = []
    
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= thresh:
                chunks[current_chunk].extend(sentence.split(" "))
            else:
                current_chunk += 1
                chunks.append(sentence.split(" "))
        else:
            chunks.append(sentence.split(" "))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])
    
    return chunks
    
@st.experimental_singleton(suppress_st_warning=True)
def gen_embeddings(embedding_model):

    '''Generate embeddings for given model'''
    
    if 'hkunlp' in embedding_model:
        
        embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model,
                                           query_instruction='Represent the Financial question for retrieving supporting paragraphs: ',
                                           embed_instruction='Represent the Financial paragraph for retrieval: ')

    else:
        
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    return embeddings
    
@st.experimental_memo(suppress_st_warning=True)
def embed_text(query,title,embedding_model,_emb_tok,_docsearch,chain_type):
    try:
        '''Embed text and generate semantic search scores'''

        title = title.split()[0].lower()
                    
        docs = _docsearch.similarity_search_with_score(query, k=3)

        if chain_type == 'Normal':

            docs = [d[0] for d in docs]

            PROMPT = PromptTemplate(template=template, 
                                    input_variables=["summaries", "question"],
                                    output_parser=output_parser)
            
            chain = load_qa_with_sources_chain(OpenAI(temperature=0), 
                                            chain_type="stuff", 
                                            prompt=PROMPT, 
                                            )

            answer = chain({"input_documents": docs, "question": query}, return_only_outputs=False)


        elif chain_type == 'Refined':

            docs = [d[0] for d in docs]
            
            initial_qa_prompt = PromptTemplate(
        input_variables=["context_str", "question"], template=initial_qa_template
    )
            chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine", return_refine_steps=False,
                        question_prompt=initial_qa_prompt, refine_prompt=refine_prompt)
            answer = chain({"input_documents": docs, "question": query}, return_only_outputs=False)
    except Exception as e:
        print(e)
        return e
    return answer
    
@st.experimental_singleton(suppress_st_warning=True)
def get_spacy():
    nlp = en_core_web_lg.load()
    return nlp
    
# Function to save uploaded file
def save_uploadedfile(filename, uploadedfile):
    with open(os.path.join("./temp/",filename),"wb") as f:
         f.write(uploadedfile.getbuffer())

# @st.experimental_memo(suppress_st_warning=True)
def inference(state, _asr_model, type):
    '''Convert Youtube video or Audio upload to text'''
    if type == 'upload':
        # TODO: remove
        print('upload handling...')
        #Orgin commend for CUDA "options = whisper.DecodingOptions()"
        results = _asr_model.transcribe("./temp/audio.mp3")
        print(results)
        # # load audio and pad/trim it to fit 30 seconds
        # audio = whisper.load_audio("audio.mp3")
        # audio = whisper.pad_or_trim(audio)

        # # make log-Mel spectrogram and move to the same device as the model
        # mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # # detect the spoken language
        # _, probs = model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # # decode the audio
        # options = whisper.DecodingOptions()
        # result = whisper.decode(model, mel, options)

        # results = _asr_model.transcribe(state, task='transcribe', language='en')
        return results['text'], "Transcribed Audio"

    elif type == 'url':
        if validators.url(state):
            yt = YouTube(state)
            title = yt.title
            path = yt.streams.filter(only_audio=True)[0].download(filename="audio.mp4")
            results = _asr_model.transcribe(path, task='transcribe', language='en')
        else:
            return "no URL existed!"
        return results['text'], yt.title
 
      
@st.experimental_memo(suppress_st_warning=True)
def sentiment_pipe(audio_text):
    '''Determine the sentiment of the text'''
    
    audio_sentences = chunk_long_text(audio_text,50,1,1)
    audio_sentiment = sent_pipe(audio_sentences)
    
    return audio_sentiment, audio_sentences    

@st.experimental_memo(suppress_st_warning=True)
def summarize_text(text_to_summarize,max_len,min_len):
    '''Summarize text with HF model'''
    
    summarized_text = sum_pipe(text_to_summarize,max_length=max_len,min_length=min_len,clean_up_tokenization_spaces=True,no_repeat_ngram_size=4,
           encoder_no_repeat_ngram_size=3,
           repetition_penalty=3.5,
           num_beams=4,
           early_stopping=True)
    summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
     
    return summarized_text
     
@st.experimental_memo(suppress_st_warning=True)
def clean_text(text):
    '''Clean all text'''

    text = text.encode("ascii", "ignore").decode()  # unicode
    text = re.sub(r"https*\S+", " ", text)  # url
    text = re.sub(r"@\S+", " ", text)  # mentions
    text = re.sub(r"#\S+", " ", text)  # hastags
    text = re.sub(r"\s{2,}", " ", text)  # over spaces
    
    return text
       
@st.experimental_memo(suppress_st_warning=True)
def chunk_long_text(text,threshold,window_size=3,stride=2):
    '''Preprocess text and chunk for sentiment analysis'''
    
    #Convert cleaned text into sentences
    sentences = sent_tokenize(text)
    out = []

    #Limit the length of each sentence to a threshold
    for chunk in sentences:
        if len(chunk.split()) < threshold:
            out.append(chunk)
        else:
            words = chunk.split()
            num = int(len(words)/threshold)
            for i in range(0,num*threshold+1,threshold):
                out.append(' '.join(words[i:threshold+i]))
    
    passages = []
    
    #Combine sentences into a window of size window_size
    for paragraph in [out]:
        for start_idx in range(0, len(paragraph), stride):
            end_idx = min(start_idx+window_size, len(paragraph))
            passages.append(" ".join(paragraph[start_idx:end_idx]))
            
    return passages   

    
def summary_downloader(raw_text):
    
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "new_text_file_{}_.txt".format(time_str)
	st.markdown("#### Download Summary as a File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click to Download!!</a>'
	st.markdown(href,unsafe_allow_html=True)

@st.experimental_memo(suppress_st_warning=True) 	
def get_all_entities_per_sentence(text):
    doc = nlp(''.join(text))

    sentences = list(doc.sents)

    entities_all_sentences = []
    for sentence in sentences:
        entities_this_sentence = []

        # SPACY ENTITIES
        for entity in sentence.ents:
            entities_this_sentence.append(str(entity))

        # XLM ENTITIES
        entities_xlm = [entity["word"] for entity in ner_pipe(str(sentence))]
        for entity in entities_xlm:
            entities_this_sentence.append(str(entity))

        entities_all_sentences.append(entities_this_sentence)

    return entities_all_sentences
 
@st.experimental_memo(suppress_st_warning=True)    
def get_all_entities(text):
    all_entities_per_sentence = get_all_entities_per_sentence(text)
    return list(itertools.chain.from_iterable(all_entities_per_sentence))

@st.experimental_memo(suppress_st_warning=True)    
def get_and_compare_entities(article_content,summary_output):
    
    all_entities_per_sentence = get_all_entities_per_sentence(article_content)
    entities_article = list(itertools.chain.from_iterable(all_entities_per_sentence))
   
    all_entities_per_sentence = get_all_entities_per_sentence(summary_output)
    entities_summary = list(itertools.chain.from_iterable(all_entities_per_sentence))
   
    matched_entities = []
    unmatched_entities = []
    for entity in entities_summary:
        if any(entity.lower() in substring_entity.lower() for substring_entity in entities_article):
            matched_entities.append(entity)
        elif any(
                np.inner(sbert.encode(entity, show_progress_bar=False),
                         sbert.encode(art_entity, show_progress_bar=False)) > 0.9 for
                art_entity in entities_article):
            matched_entities.append(entity)
        else:
            unmatched_entities.append(entity)

    matched_entities = list(dict.fromkeys(matched_entities))
    unmatched_entities = list(dict.fromkeys(unmatched_entities))

    matched_entities_to_remove = []
    unmatched_entities_to_remove = []

    for entity in matched_entities:
        for substring_entity in matched_entities:
            if entity != substring_entity and entity.lower() in substring_entity.lower():
                matched_entities_to_remove.append(entity)

    for entity in unmatched_entities:
        for substring_entity in unmatched_entities:
            if entity != substring_entity and entity.lower() in substring_entity.lower():
                unmatched_entities_to_remove.append(entity)

    matched_entities_to_remove = list(dict.fromkeys(matched_entities_to_remove))
    unmatched_entities_to_remove = list(dict.fromkeys(unmatched_entities_to_remove))

    for entity in matched_entities_to_remove:
        matched_entities.remove(entity)
    for entity in unmatched_entities_to_remove:
        unmatched_entities.remove(entity)

    return matched_entities, unmatched_entities

@st.experimental_memo(suppress_st_warning=True) 
def highlight_entities(article_content,summary_output):
   
    markdown_start_red = "<mark class=\"entity\" style=\"background: rgb(238, 135, 135);\">"
    markdown_start_green = "<mark class=\"entity\" style=\"background: rgb(121, 236, 121);\">"
    markdown_end = "</mark>"

    matched_entities, unmatched_entities = get_and_compare_entities(article_content,summary_output)
    
    print(summary_output)

    for entity in matched_entities:
        summary_output = re.sub(f'({entity})(?![^rgb\(]*\))',markdown_start_green + entity + markdown_end,summary_output)

    for entity in unmatched_entities:
        summary_output = re.sub(f'({entity})(?![^rgb\(]*\))',markdown_start_red + entity + markdown_end,summary_output)
    
    print("")
    print(summary_output)
    
    print("")
    print(summary_output)
    
    soup = BeautifulSoup(summary_output, features="html.parser")

    return HTML_WRAPPER.format(soup)
    
    
def display_df_as_table(model,top_k,score='score'):
    '''Display the df with text and scores as a table'''
    
    df = pd.DataFrame([(hit[score],passages[hit['corpus_id']]) for hit in model[0:top_k]],columns=['Score','Text'])
    df['Score'] = round(df['Score'],2)
    
    return df   

      
def make_spans(text,results):
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    facts_spans = []
    facts_spans = list(zip(sent_tokenizer(text),results_list))
    return facts_spans

##Fiscal Sentiment by Sentence
def fin_ext(text):
    results = remote_clx(sent_tokenizer(text))
    return make_spans(text,results)

## Knowledge Graphs code

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

def from_text_to_kb(text, model, tokenizer, article_url, span_length=128, article_title=None,
                    article_publish_date=None, verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                article_url: {
                    "spans": [spans_boundaries[current_span_index]]
                }
            }
            kb.add_relation(relation, article_title, article_publish_date)
        i += 1

    return kb

def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

def from_url_to_kb(url, model, tokenizer):
    article = get_article(url)
    config = {
        "article_title": article.title,
        "article_publish_date": article.publish_date
    }
    kb = from_text_to_kb(article.text, model, tokenizer, article.url, **config)
    return kb

def get_news_links(query, lang="en", region="US", pages=1):
    googlenews = GoogleNews(lang=lang, region=region)
    googlenews.search(query)
    all_urls = []
    for page in range(pages):
        googlenews.get_page(page)
        all_urls += googlenews.get_links()
    return list(set(all_urls))

def from_urls_to_kb(urls, model, tokenizer, verbose=False):
    kb = KB()
    if verbose:
        print(f"{len(urls)} links to visit")
    for url in urls:
        if verbose:
            print(f"Visiting {url}...")
        try:
            kb_url = from_url_to_kb(url, model, tokenizer)
            kb.merge_with_kb(kb_url)
        except ArticleException:
            if verbose:
                print(f"  Couldn't download article at url {url}")
    return kb

def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="700px", height="700px")

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])

    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename)

def save_kb(kb, filename):
    with open(filename, "wb") as f:
        pickle.dump(kb, f)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'KB':
            return KB
        return super().find_class(module, name)

def load_kb(filename):
    res = None
    with open(filename, "rb") as f:
        res = CustomUnpickler(f).load()
    return res

class KB():
    def __init__(self):
        self.entities = {} # { entity_title: {...} }
        self.relations = [] # [ head: entity_title, type: ..., tail: entity_title,
          # meta: { article_url: { spans: [...] } } ]
        self.sources = {} # { article_url: {...} }

    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            article_url = list(r["meta"].keys())[0]
            source_data = kb2.sources[article_url]
            self.add_relation(r, source_data["article_title"],
                              source_data["article_publish_date"])

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r2):
        r1 = [r for r in self.relations
              if self.are_relations_equal(r2, r)][0]

        # if different article
        article_url = list(r2["meta"].keys())[0]
        if article_url not in r1["meta"]:
            r1["meta"][article_url] = r2["meta"][article_url]

        # if existing article
        else:
            spans_to_add = [span for span in r2["meta"][article_url]["spans"]
                            if span not in r1["meta"][article_url]["spans"]]
            r1["meta"][article_url]["spans"] += spans_to_add

    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r, article_title, article_publish_date):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # add source if not in kb
        article_url = list(r["meta"].keys())[0]
        if article_url not in self.sources:
            self.sources[article_url] = {
                "article_title": article_title,
                "article_publish_date": article_publish_date
            }

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def get_textual_representation(self):
        res = ""
        res += "### Entities\n"
        for e in self.entities.items():
            # shorten summary
            e_temp = (e[0], {k:(v[:100] + "..." if k == "summary" else v) for k,v in e[1].items()})
            res += f"- {e_temp}\n"
        res += "\n"
        res += "### Relations\n"
        for r in self.relations:
            res += f"- {r}\n"
        res += "\n"
        res += "### Sources\n"
        for s in self.sources.items():
            res += f"- {s}\n"
        return res
            
def save_network_html(kb, filename="knowledge_network.html"):
    # create network
    net = Network(directed=True, width="700px", height="700px", bgcolor="#eeeeee")

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])
        
    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename)

nlp = get_spacy()    

sent_pipe, sum_pipe, ner_pipe, cross_encoder, kg_model, kg_tokenizer, emb_tokenizer, sbert  = load_models()
