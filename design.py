import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Custom CSS for the attractive front end
custom_css = """
<style>
body {
    background-color: #f0f0f0;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 600px;
    margin: auto;
    padding: 20px;
    border-radius: 10px;
    background-color: #fff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.title {
    text-align: center;
    font-size: 28px;
    color: #333;
    margin-bottom: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 20px;
}

.upload-box {
    background-color: #f9f9f9;
    border: 2px dashed #ddd;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
}

.upload-box:hover {
    border-color: #aaa;
}

.upload-box p {
    color: #777;
    font-size: 16px;
}

.upload-box input[type="file"] {
    display: none;
}

.upload-btn {
    background-color: #007bff;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}

.upload-btn:hover {
    background-color: #0056b3;
}

.predicted-label {
    font-size: 24px;
    margin-top: 20px;
    text-align: center;
}

.audio-player {
    margin-top: 20px;
}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

@st.cache
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
    st.markdown("<h1 class='title'>SASTRA MAJOR Project</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>Voice Genre Prediction</h2>", unsafe_allow_html=True)
    st.write("This app predicts whether the uploaded audio file is bonafide or spoof.")

    model_path = "cnn_audio.h5"

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        if model_path:
            model = load_audio_model(model_path)
            genre_mapping = {0: "spoof", 1: "bonafide"}
            predicted_voice = predict_voice(model, uploaded_file, genre_mapping)
            st.markdown(f"<p class='predicted-label'>Predicted label: {predicted_voice}</p>", unsafe_allow_html=True)
            
            # Play the uploaded audio file
            st.audio(uploaded_file, format='audio/wav', class_='audio-player')
        else:
            st.write("Please provide the model path.")

if __name__ == "__main__":
    main()
