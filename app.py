import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Custom CSS for the hacker-themed background
hacker_css = f"""
body {{
    background-image: url('file:///E:/Project/New/hacker-5471975_1920.png');
    background-size: cover;
    color: #0F0;
}}

.section-header {{
    background-color: rgba(0, 0, 0, 0.7);
    padding: 10px;
    border-radius: 10px;
}}
"""

# Apply the custom CSS
st.markdown(f'<style>{hacker_css}</style>', unsafe_allow_html=True)

def load_audio_model(model_path):
    return load_model(model_path)

def predict_voice(model, audio_data, genre_mapping):
    signal, sample_rate = librosa.load(audio_data, sr=22050)

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # Resize MFCC to the appropriate size
    mfcc = np.resize(mfcc, (130, 13, 1))

    # Reshape MFCC
    mfcc = mfcc[np.newaxis, ...]

    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)

    genre_label = genre_mapping[predicted_index[0]]
    return genre_label

def main():
    # st.title("SASTRA MAJOR Project")
    st.title("VOICE GENRE PREDICTION")
    st.write("This app predicts whether the uploaded audio file is bonafide or spoof.")

    model_path = "cnn_audio.h5"

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        if model_path:
            model = load_audio_model(model_path)
            genre_mapping = {0: "spoof", 1: "bonafide"}
            predicted_voice = predict_voice(model, uploaded_file, genre_mapping)
            st.write("Predicted label:", predicted_voice)
            
            # Play the uploaded audio file
            st.audio(uploaded_file, format='audio/wav')
        else:
            st.write("Please provide the model path.")

if __name__ == "__main__":
    main()
