from pydub import AudioSegment
from scipy import signal
from scipy.io.wavfile import read, write
import numpy as np
from numpy import array, int16

class audio_processing(object):
    "Add effects on the received mp3 file, and creates a wav file with the choosen effects"
    __slots__ = ("audio_data", "sample_freq")

    def __init__(self, input_audio_path):
        self.sample_freq, self.audio_data = read(input_audio_path)
        self.audio_data = audio_processing.convert_to_mono_audio(self.audio_data)

    def save_to_file(self, output_path):
        "Writes a wav file with the effects"
        write(output_path, self.sample_freq, array(self.audio_data, dtype = int16))

    def set_reverse(self):
        "Reverses the audio"
        self.audio_data = self.audio_data[::-1]

    def set_volume(self, level):
        "Sets the overall volume of the audio via floating-point factor"
        output_audio = np.zeros(len(self.audio_data))
        for count, e in enumerate(self.audio_data):
            output_audio[count] = e * level
        self.audio_data = output_audio

    def set_speed(self, speed_factor):
        "Sets the speed of the audio by a floating-point factor"
        sound_index = np.round(np.arange(0, len(self.audio_data), speed_factor))
        self.audio_data = self.audio_data[sound_index[sound_index < len(self.audio_data)].astype(int)]

    def set_lowpass(self, cutoff_low, order = 5):
        "Applies a low-pass filter"
        nyquist = self.sample_freq / 2.0
        cutoff = cutoff_low / nyquist
        x, y = signal.butter(order, cutoff, btype = "lowpass", analog = False)
        self.audio_data = signal.filtfilt(x, y, self.audio_data, axis = 0)

    def set_highpass(self, cutoff_high, order = 5):
        "Applies a high-pass filter"
        nyquist = self.sample_freq / 2.0
        cutoff = cutoff_high / nyquist
        x, y = signal.butter(order, cutoff, btype = "highpass", analog = False)
        self.audio_data = signal.filtfilt(x, y, self.audio_data, axis = 0)

    def set_bandpass(self, cutoff_low, cutoff_high, order = 5):
        "Applies a band-pass filter"
        cutoff = np.zeros(2)
        nyquist = self.sample_freq / 2.0
        cutoff[0] = cutoff_low / nyquist
        cutoff[1] = cutoff_high / nyquist
        x, y = signal.butter(order, cutoff, btype = "bandpass", analog = False)
        self.audio_data = signal.filtfilt(x, y, self.audio_data, axis = 0)

    def set_echo(self, delay):
        "Applies an echo effect after x (floating-point number) seconds from the start"
        output_audio = np.zeros(len(self.audio_data))
        output_delay = delay * self.sample_freq
        for count, e in enumerate(self.audio_data):
            output_audio[count] = e + self.audio_data[count - int(output_delay)]
        self.audio_data = output_audio

    def set_reverb(self, type=None):
        "Convolve the audio with the variable impulse_response to get the vector with reverb applied"
        impulse_response = np.genfromtxt("impulse_hall.csv", dtype = "float32")
        if type == "church":
            impulse_response = np.genfromtxt("impulse_church.csv", dtype = "float32")
        self.audio_data = signal.fftconvolve(self.audio_data, impulse_response)

    def set_audio_pitch(self, n, window_size=2**13, h=2**11):
        "Sets the pitch of the audio to a certain threshold"
        factor = 2 ** (1.0 * n / 12.0)
        self._set_stretch(1.0 / factor, window_size, h)
        self.audio_data = self.audio_data[window_size:]
        self.set_speed(factor)

    def _set_stretch(self, factor, window_size, h):
        phase = np.zeros(window_size)
        hanning_window = np.hanning(window_size)
        result = np.zeros(int(len(self.audio_data) / factor + window_size))

        for i in np.arange(0, len(self.audio_data) - (window_size + h), h*factor):
            # Two potentially overlapping subarrays
            a1 = self.audio_data[int(i): int(i + window_size)]
            a2 = self.audio_data[int(i + h): int(i + window_size + h)]

            # The spectra of these arrays
            s1 = np.fft.fft(hanning_window * a1)
            s2 = np.fft.fft(hanning_window * a2)

            # Rephase all frequencies
            phase = (phase + np.angle(s2/s1)) % 2*np.pi

            a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))
            i2 = int(i / factor)
            result[i2: i2 + window_size] += hanning_window*a2_rephased.real

        # normalize (16bit)
        result = ((2 ** (16 - 4)) * result/result.max())
        self.audio_data = result.astype("int16")

    def convert_to_mono_audio(input_audio):
        "Returns a numpy array that represents the mono version of a stereo input"
        output_audio = []
        temp_audio = input_audio.astype(float)
        for i in temp_audio:
            frac, whole = np.modf(i)
            output_audio.append((whole / 2) + (frac / 2))
        return np.array(output_audio, dtype = "int16")

    def data_augmentation_audios(file_name):
        sound = AudioSegment.from_mp3(file_name + ".mp3")
        sound = sound.set_channels(1)
        sound.export(file_name + ".wav", format = "wav")

        speed = audio_processing(file_name + ".wav")
        speed.set_speed(1.2)
        speed.save_to_file(file_name + "_speed.wav")

        lowpass = audio_processing(file_name + ".wav")
        lowpass.set_lowpass(1250)
        lowpass.save_to_file(file_name + "_lowpass.wav")

        highpass = audio_processing(file_name + ".wav")
        highpass.set_highpass(750)
        highpass.save_to_file(file_name + "_highpass.wav")

        bandpass = audio_processing(file_name + ".wav")
        bandpass.set_bandpass(500, 1950)
        bandpass.save_to_file(file_name + "_bandpass.wav")

        echo = audio_processing(file_name + ".wav")
        echo.set_echo(0.03)
        echo.save_to_file(file_name + "_echo.wav")

        radio = audio_processing(file_name + ".wav")
        radio.set_highpass(2000)
        radio.set_volume(3.0)
        radio.set_bandpass(50, 2600)
        radio.set_volume(2.0)
        radio.save_to_file(file_name + "_radio.wav")

        robotic = audio_processing(file_name + ".wav")
        robotic.set_volume(1.05)
        robotic.set_echo(0.01)
        robotic.set_bandpass(300, 4000)
        robotic.save_to_file(file_name + "_robotic.wav")

        darth_vader = audio_processing(file_name + ".wav")
        darth_vader.set_speed(.9)
        darth_vader.set_echo(0.02)
        darth_vader.set_lowpass(2500)
        darth_vader.save_to_file(file_name + "_darth_vader.wav")

    def data_augmentation_texts(file_name, write_this):
        effects_list= ['_speed', '_lowpass', '_highpass', '_bandpass', '_echo', '_radio', '_robotic', '_darth_vader']
        for i in effects_list:
            file = open(file_name + i + ".txt","w+")
            file.write(write_this)
            file.close()
        file = open(file_name + ".txt","w+")
        file.write(write_this)
        file.close()

audio_processing.data_augmentation_audios("ahri_jogo")
audio_processing.data_augmentation_texts("ahri_jogo", "se você deseja jogar comigo é melhor ter certeza de que conhece o jogo")
