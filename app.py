import streamlit as st
import librosa
import numpy as np
from keras.models import load_model
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import tensorflow as tf
# Load the ML model
# model = joblib.load('my_model.joblib')
model = tf.keras.models.load_model('NewModel.h5')


# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fearful',
                  'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to extract audio features


def extract_features(data, sample_rate):
    # # ZCR
    # zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

    # # Chroma_stft
    # stft = np.abs(librosa.stft(data))
    # chroma_stft = np.mean(librosa.feature.chroma_stft(
    #     S=stft, sr=sample_rate).T, axis=0)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(
        y=data, n_mfcc=40, sr=sample_rate).T, axis=0)

    # # Root Mean Square Value
    # rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    # # MelSpectrogram
    # mel = np.mean(librosa.feature.melspectrogram(
    #     y=data, sr=sample_rate).T, axis=0)

    result = mfcc
    return result

# Function to make predictions using the ML model


def make_predictions(features):
    # Preprocess the features, if necessary
    # ...

    # Make predictions using the loaded model
    preds = model.predict(features[None])
    # Get the index of the highest probability class
    preds = np.argmax(preds, axis=1)
    return preds

# Function to process the audio file using the loaded model
# def process_audio(file):
#     data, sample_rate = librosa.load(file)
#     features = extract_features(data, sample_rate)
#     # features = np.expand_dims(features, axis=0)  # Expand dimensions to match the model's input shape
#     predictions = make_predictions(features)
#     return predictions


def process_audio(file):
    data, sample_rate = librosa.load(file)
    # write("Audio Data Shape:", data.shape)
   # write("Sample Rate:", sample_rate)
    features = extract_features(data, sample_rate)
    # features = np.expand_dims(features, axis=0)  # Convert features to a batch format
    predictions = make_predictions(features)
    return predictions


def recAudio():
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file

    st.write("Audio recorded")

# Streamlit web app
# Streamlit web app


def main():
    st.set_page_config(page_title="Audio Processing App")
    st.title("Emotion Detection App")

    # Placeholder for predictions
    predictions_placeholder = st.empty()

    # Navigation
    pages = {
        "Home": home_page,
        "About Us": about_us_page
    }

    # Initialize session state
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"

    # Display the navigation bar
    navigation_bar(pages, predictions_placeholder)

# Navigation bar for page selection


def navigation_bar(pages, predictions_placeholder):
    selected_page = st.session_state.selected_page

    st.sidebar.title("Navigation")
    if st.sidebar.button("Home", key="home"):
        st.session_state.selected_page = "Home"
    if st.sidebar.button("About Us", key="about"):
        st.session_state.selected_page = "About Us"
    selected_page = st.session_state.selected_page

    # Display the selected page
    pages[selected_page](predictions_placeholder)


# Home page with file upload feature
def home_page(predictions_placeholder):
    predictions_placeholder.text("Predicted Emotion: ")
    st.title("Home")
    with st.form("my_form"):
        uploaded_file = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3"], key="up")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if uploaded_file is not None:
                predictions = process_audio(file=uploaded_file)
                st.success("Audio file processed successfully!")

                # Convert predicted class index to emotion label
                predicted_emotion = np.array(emotion_labels)[
                    predictions[0].astype(int)]

                # Update the predictions placeholder with the results
                predictions_placeholder.write(
                    "Predicted Emotion: " + predicted_emotion)
            else:
                st.warning("Please upload an audio file.")
        st.subheader("Record Audio")
        record_button = st.form_submit_button("Start Recording")

        if record_button:
            st.write("recording started")
            recAudio()
            predictions = process_audio('output.wav')
            os.remove('output.wav')
            st.success("Audio file processed successfully!")

            # Convert predicted class index to emotion label
            predicted_emotion = np.array(emotion_labels)[
                predictions[0].astype(int)]

            # Update the predictions placeholder with the results
            predictions_placeholder.write(
                "Predicted Emotion: " + predicted_emotion)


# About Us page
def about_us_page(predictions_placeholder):
    st.title("About Us")
    st.markdown(
        """
        This web app was developed by the following team members:

        **Ayush Awasthi**
        - Role: Team Leader

        **Atharva Nanoti**
        - Role: Team Member

        **Shaurya Khetarpal**
        - Role: Team Member

        **Jasmer Singh Sanjotra**
        - Role: Team Member

        """
    )


if __name__ == '__main__':
    main()
