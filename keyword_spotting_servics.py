import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050 #equal to 1 sec

class _keyword_spotting_Servics:
    """Singleton class for keyword spotting inference with trained models.

    :param model: Trained model
    """

    model = None
    _mapping = [
         "1a1",
        "1a2",
        "1a3",
        "3a1",
        "3a2",
        "3a3",
        "4a1",
        "4a2",
        "4a3",
        "5a1",
        "5a2",
        "5a3",
        "6a1",
        "6a2",
        "6a3",
        "6aa1",
        "6aa2",
        "6aa3",
        "7a1",
        "7a2",
        "7a3",
        "8a1",
        "8a2",
        "8a3",
        "9a1",
        "9a2",
        "9a3",
        "a1",
        "a2",
        "a3",
        "b1",
        "b2",
        "b3",
        "da1",
        "da2",
        "da3",
        "fa1",
        "fa2",
        "fa3",
        "gem1",
        "gem2",
        "gem3",
        "gha1",
        "gha2",
        "gha3",
        "ha1",
        "ha2",
        "ha3",
        "ka1",
        "ka2",
        "ka3",
        "la1",
        "la2",
        "la3",
        "mem1",
        "mem2",
        "mem3",
        "non1",
        "non2",
        "non3",
        "ra1",
        "ra2",
        "ra3",
        "sen1",
        "sen2",
        "sen3",
        "shen1",
        "shen2",
        "shen3",
        "ta1",
        "ta2",
        "ta3",
        "tha1",
        "tha2",
        "tha3",
        "waw1",
        "waw2",
        "waw3",
        "ya1",
        "ya2",
        "ya3",
        "za1",
        "za2",
        "za3"
    ]
    _instance = None


    def predict(self, file_path):
        """

        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.

        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples

        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.

    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _keyword_spotting_Servics._instance is None:
        _keyword_spotting_Servics._instance = _keyword_spotting_Servics()
        _keyword_spotting_Servics.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _keyword_spotting_Servics._instance




if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict(r"C:\Users\Razer\OneDrive\Desktop\ArabicDataSet\a1\a1 (1).wav")
   # keyword = kss.predict("testAudioSample\ao (2).wav")
    print(keyword)