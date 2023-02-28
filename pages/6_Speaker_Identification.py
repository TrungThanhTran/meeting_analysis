import streamlit as st
from pyvis.network import Network
from functions import *
import streamlit.components.v1 as components
import pickle, math
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from torchaudio.sox_effects import apply_effects_tensor
import os
import torch
import pydub
import torchaudio
from PIL import Image
from glob import glob

# replace logo
image_directory = "data/logo/logo.png"
image_logo = Image.open(image_directory)
st.set_page_config(page_title="Capturia",page_icon=image_logo)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(file_name):
    audio = pydub.AudioSegment.from_file(file_name)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate

THRESHOLD = 0.85

EFFECTS = [
    ["remix", "-"],
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-1.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    ["trim", "0", "10"],
]

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Speaker Identification")
si_model_options = ["microsoft/wavlm-base-plus-sv"]

si_model_name = st.sidebar.selectbox("Whisper Model Options", options=si_model_options, key='sibox')
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

st.markdown("## Speaker Identification")

def feature_extract(path):
    wav, sr = load_audio(path)
    print(wav, wav.shape, wav.dtype)
    wav, _ = apply_effects_tensor(torch.tensor(wav).unsqueeze(0), sr, EFFECTS)
    input1 = feature_extractor(wav.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
         emb = model(input1).embeddings
    emb = torch.nn.functional.normalize(emb, dim=-1).cpu()
    return emb

def similarity_fn(emb1, emb2):
    similarity = cosine_sim(emb1, emb2).numpy()[0]
    output = similarity * 100

    return output

# Upload sample file user
model, feature_extractor = load_si_model(si_model_name)

with st.expander("Upload sample voice"):
    upload_registration = st.file_uploader("",key="upload_registration")
    user_name = st.text_input('speaker name')

    confirm_reg = st.button('Register')
    if confirm_reg:
        with st.spinner("Uploading registration..."):
            if upload_registration:
                if not os.path.exists(f"./temp/registration/{user_name}"):
                    os.mkdir(f"./temp/registration/{user_name}")
                    
                with open(f"./temp/registration/{user_name}/{user_name}.mp3","wb") as f:
                    f.write(upload_registration.getbuffer())

st.markdown("<br><br>",  unsafe_allow_html=True)

st.markdown("### Upload check voice")
upload_check = st.file_uploader("",key="upload_check")
if upload_check:
    check_name = upload_check.name

    if not os.path.exists(f"./temp/check"):
        os.mkdir(f"./temp/check")
                
    with open(f"./temp/check/{check_name}","wb") as f:
        f.write(upload_check.getbuffer())

confirm_check = st.button('Check ID')                 
if confirm_check:
    with st.spinner("Recognizing..."):
        folders = glob('./temp/registration/*')
        dict_names = {}
        # Make dict emb of all reg
        for user_name in folders:
            files = glob(os.path.join(user_name, '*'))
            reg_emb = feature_extract(files[0])
            dict_names[user_name] = reg_emb       
        
        # upload check get embe
        _emb = feature_extract(f'./temp/check/{check_name}')
    
        # compare
        max_sim = 0
        match_name = ""     
        for _name, emb_reg in zip(dict_names.keys(), dict_names.values()):
            _sim = similarity_fn(_emb, emb_reg)
            if _sim > max_sim:
                match_name = _name
                max_sim = _sim
            
        if max_sim > THRESHOLD:    
            output = f"""
                    <div class="container">
                        <div class="row"><h1 style="text-align: center">The speakers are {match_name.split('/')[-1].split('.')[0]}</h1></div>
                        <div class="row"><h1 class="display-1 text-success" style="text-align: center">{int(max_sim)}%</h1></div>
                        <div class="row"><h1 style="text-align: center">similar</h1></div>
                        <div class="row"><h1 class="text-success" style="text-align: center">Welcome, human!</h1></div>
                    </div>
                """
        else:
            output = f"""
                    <div class="row"><h1 class="text-success" style="text-align: center">Sorry, can't recognize!</h1></div>
                    """

        st.markdown(output, unsafe_allow_html=True)
                