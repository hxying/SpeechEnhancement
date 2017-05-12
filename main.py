import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import load_model
from scipy.special import expit


class SpeechEnhancement:
    def __init__(self, batch_size=128, nb_epoch=5, lstm_units=4096, n_mfcc=128, time_fft=25, time_window=3000, time_mask=50, mask_type="binary"):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.lstm_units = lstm_units
        self.n_mfcc = n_mfcc
        self.time_fft = time_fft
        self.time_window = time_window
        self.time_mask = time_mask
        self.mask_type = mask_type

        self.model = None

    def raw_train_data(self, speech_file, noise_file, augmentation=True):  # return raw speech data and mixed data.
        ys, srs = librosa.load(speech_file)
        yn, srn = librosa.load(noise_file)
        if augmentation:
            total = 60000000
            ys = np.tile(ys, int(total / len(ys)))
            yn = np.tile(yn, int(total / len(yn)))
            length = min(len(ys), len(yn))
            step = int(length / 10)
            j = 1.2
            for i in range(0, length, step):
                ys[i:i+step] *= j
                j -= 0.1
        length = min(len(ys), len(yn))
        ys = ys[:length]
        yn = yn[:length]
        ymix = ys + yn
        return ys, yn, ymix, srs

    def mfcc(self, y, sr):
        n_mfcc = self.n_mfcc
        time_fft = self.time_fft
        time_window = self.time_window
        hop_length = int(time_fft / 1000 * sr)
        n_fft = hop_length * 4

        # MFCC and its derivatives.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_input = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)

        # Sliding window to generate train samples.
        window_size = int(time_window / time_fft)  # number of mfcc frames in a window.
        samples = mfcc_input.shape[1] - window_size + 1
        x_data = np.empty([samples, mfcc_input.shape[0], window_size])
        for i in range(samples):
            x_data[i] = mfcc_input[:, i:i + window_size]
        x_data = np.swapaxes(x_data, 1, 2)

        return x_data

    def train_data(self, ys, yn, ymix, sr):
        time_fft = self.time_fft
        time_window = self.time_window
        time_mask = self.time_mask
        hop_length = int(time_fft / 1000 * sr)
        n_fft = hop_length * 4

        x_data = self.mfcc(ymix, sr)
        window_size = int(time_window / time_fft)  # number of mfcc frames in a window.
        samples = x_data.shape[0]

        # Generate ideal binary mask. 0 if noise > speech, 1 if speech > noise.
        stft_s = librosa.core.stft(ys, n_fft=n_fft, hop_length=hop_length)
        stft_s_db = librosa.core.amplitude_to_db(stft_s)
        stft_n = librosa.core.stft(yn, n_fft=n_fft, hop_length=hop_length)
        stft_n_db = librosa.core.amplitude_to_db(stft_n)
        if self.mask_type == "binary":
            mask = (stft_s_db > stft_n_db) * 1  # Ideal binary mask.
        else:  # ratio or restricted_ratio.
            stft_mix = librosa.core.stft(ymix, n_fft=n_fft, hop_length=hop_length)
            stft_mix_m = np.abs(stft_mix)
            stft_s_m = np.abs(stft_s)
            mask = np.divide(stft_s_m, stft_mix_m)
            if self.mask_type == "restricted_ratio":
                mask = expit(mask)
        mask_size = int(time_mask / time_fft)
        y_data = np.empty([samples, mask.shape[0], mask_size])
        for i in range(samples):
            y_data[i] = mask[:, i + window_size - mask_size:i + window_size]  # Mask is at the end of the window.
        y_data = y_data.reshape((y_data.shape[0], -1))

        return x_data, y_data, mask

    def train(self, speech_file, noise_file, save=None):
        ys, yn, ymix, sr = self.raw_train_data(speech_file, noise_file)
        split = int(len(ys) * 0.9)
        x_train, y_train, mask_train = self.train_data(ys[:split], yn[:split], ymix[:split], sr)
        x_validate, y_validate, mask_validate = self.train_data(ys[split:], yn[split:], ymix[split:], sr)
        batch_size = self.batch_size
        nb_epoch = self.nb_epoch
        units = self.lstm_units

        model = self.model
        if model is None:
            print('Building model...')
            model = Sequential()
            model.add(LSTM(units, input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(Dense(2048))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(y_train.shape[1]))
            if self.mask_type == "binary" or self.mask_type == "restricted_ratio":
                model.add(Activation('sigmoid'))

            model.compile(optimizer='rmsprop', loss='mse')

        print('Training...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_validate, y_validate))
        if save is not None:
            model.save(save)
        self.model = model
        return model

    def recover(self, mix_file, load=None):
        ymix, sr = librosa.load(mix_file)
        if load is not None:
            self.model = load_model(load)
        model = self.model
        hop_length = int(self.time_fft / 1000 * sr)
        n_fft = hop_length * 4
        stft_mix = librosa.core.stft(ymix, n_fft=n_fft, hop_length=hop_length)

        window_size = int(self.time_window / self.time_fft)
        mask_size = int(self.time_mask / self.time_fft)
        mask_predict = np.ones(stft_mix.shape)

        x_test = self.mfcc(ymix, sr)
        y_predict = model.predict(x_test, batch_size=self.batch_size)
        if self.mask_type == "binary":
            y_predict = (y_predict > 0.5) * 1
        y_predict = y_predict.reshape((y_predict.shape[0], stft_mix.shape[0], mask_size))

        j = 0
        for i in range(window_size - mask_size, mask_predict.shape[1], mask_size):
            if j >= y_predict.shape[0]:
                break
            mask_predict[:, i:i + mask_size] = y_predict[j]
            j += mask_size

        stft_recover = np.multiply(stft_mix, mask_predict)
        recover = librosa.core.istft(stft_recover, hop_length=hop_length)
        return recover, sr, mask_predict


def main():
    speech_file = "train_speech.wav"
    noise_file = "train_noise.m4a"
    test_file = "test_mix.wav"

    mask_type = ["binary", "ratio", "restricted_ratio"]
    filename = ["recover_binary", "recover_ratio", "recover_restricted_ratio"]
    for i in range(2, 3):
        model = None  # free up memory.
        model = SpeechEnhancement(nb_epoch=5, mask_type=mask_type[i])
        model.train(speech_file, noise_file, filename[i] + "_model.h5")
        recover, sr, mask = model.recover(test_file)
        np.save(filename[i] + "_mask.npy", mask)
        librosa.output.write_wav(filename[i] + ".wav", recover, sr)


if __name__ == '__main__':
    main()
