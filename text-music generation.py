from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration: int):
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output

def save_audio(samples, counter):
    sample_rate = 32000
    save_path = "audio_output/"
    
    audio_paths = []
    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"music{counter}_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
        audio_paths.append(audio_path)
    return audio_paths

def get_binary_file_downloader_html(bin_files, file_labels):
    hrefs = []
    for bin_file, file_label in zip(bin_files, file_labels):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        hrefs.append(href)
    return hrefs

st.set_page_config(
    page_icon= "musical_note",
    page_title= "Music Generator"
)

counter = 1 

def main():
    global counter  
    st.title("Text to Music Generator using Python")

    with st.sidebar:
        st.subheader("Available Song Genres:")
        st.write("- Pop")
        st.write("- Rock")
        st.write("- Jazz")
        st.write("- Classical")
        st.write("- Hip-hop")
        st.write("- Electronic")
        st.write("- Country")

    st.markdown("---")  # Separator between sidebar and main content

    st.write("Please enter a sentence or lyrics describing the mood or style of music you want to generate.")
    text_area = st.text_area("Enter your sentence or lyrics")

    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if st.button("Generate Music"):
        if text_area and time_slider:
            st.json({
                'Your Description': text_area,
                'Selected Time Duration (in Seconds)': time_slider
            })

            st.subheader("Generated Music")
            music_tensors = generate_music_tensors(text_area, time_slider)
            audio_filepaths = save_audio(music_tensors, counter)
            audio_file_labels = [f"Audio {i+1}" for i in range(len(music_tensors))]
            for i in range(len(music_tensors)):
                st.audio(open(audio_filepaths[i], 'rb').read(), format='audio/wav')
                st.markdown(get_binary_file_downloader_html([audio_filepaths[i]], [audio_file_labels[i]])[0], unsafe_allow_html=True)

            counter += 1  

if _name_ == "_main_":
    main()