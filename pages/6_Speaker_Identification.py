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

model_name = "microsoft/wavlm-base-plus-sv"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

footer="""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Speaker Identification")
st.markdown("## Speaker Identification")

def similarity_fn(path1, path2):
    if not (path1 and path2):
        return '<b style="color:red">ERROR: Please record audio for *both* speakers!</b>'
    
    wav1, sr1 = load_audio(path1)
    print(wav1, wav1.shape, wav1.dtype)
    wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS)
    wav2, sr2 = load_audio(path2)
    wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS)
    print(wav1.shape, wav2.shape)

    input1 = feature_extractor(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)
    input2 = feature_extractor(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        emb1 = model(input1).embeddings
        emb2 = model(input2).embeddings
    emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
    emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()
    similarity = cosine_sim(emb1, emb2).numpy()[0]

    if similarity >= THRESHOLD:
        output = similarity * 100
    else:
        output = similarity * 100

    return output

# Upload sample file user
st.write("Upload sample voice")
upload_wav = st.file_uploader("Upload a .wav sound file ",key="upload")
user_name = st.text_input('Voice name', 'Robin X')
if upload_wav:
    if not os.path.exists(os.path.join(f"./temp/registration/{user_name}")):
        os.mkdir(os.path.join(f"./temp/registration/{user_name}"))
        
    with open(os.path.join(f"./temp/registration/{user_name}/{user_name}.mp3"),"wb") as f:
        f.write(upload_wav.getbuffer())

# Get file in folder temp and user name to get file registraion

# Get embedding

# Compare with each seqment in the audio file

# Print results