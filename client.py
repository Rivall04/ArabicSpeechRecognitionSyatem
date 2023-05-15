import requests
from flask import Flask, request, jsonify
import wave
import pyaudio
from playsound import playsound



# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to   send for predicting keyword
#FILE_PATH = r"C:\Users\Razer\OneDrive\Documents\speechRecognitionApplication\testAudioSample\a1\ (1).wav"

WAVE_OUTPUT_FILENAME = "waveOutPut\output.wav"

def record():

 CHUNK = 1024
 
 FORMAT = pyaudio.paInt16

 CHANNELS = 2

 RATE = 44100

 RECORD_SECONDS = 5

 #WAVE_OUTPUT_FILENAME = "waveOutput\output.wav"


 p = pyaudio.PyAudio()


 stream = p.open(format=FORMAT,

                channels=CHANNELS,

                rate=RATE,

                input=True,

                frames_per_buffer=CHUNK)


 print("* recording")


 frames = []


 for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    data = stream.read(CHUNK)

    frames.append(data)


 print("* done recording")


 stream.stop_stream()

 stream.close()

 p.terminate()

 wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

 wf.setnchannels(CHANNELS)

 wf.setsampwidth(p.get_sample_size(FORMAT))

 wf.setframerate(RATE)

 wf.writeframes(b''.join(frames))

 wf.close()
 #return jsonify({'keyword' :record })

if __name__ == "__main__":
    # open files
    record()
    file = open(WAVE_OUTPUT_FILENAME, "rb")
     
    # package stuff to send and perform POST request
    values = {"file": (WAVE_OUTPUT_FILENAME, file, "audio/wav")}
    response = requests.post(URL, files=values)
    
    data = response.json()
    if data == data :
      playsound('sound.wav')  
    else:
      playsound
    print("Predict keyword: {}".format(data["keyword"]))
