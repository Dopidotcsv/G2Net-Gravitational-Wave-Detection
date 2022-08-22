import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d  # interpolating a 1-D function
import matplotlib.mlab as mlab  # some MATLAB commands
from src.load.get_data import *

import librosa
import librosa.display

def plot_raw_data(path,
           df,
           target,
           labels = ('LIGO Hanford', 'LIGO Livingston', 'Virgo')
):
    sample_id = df[df['target'] == target].sample(random_state=42)['id'].values[0]
    sample_id = int(sample_id)
    training_files = glob(path)
    data = np.load(training_files[sample_id])
    fig, ax = plt.subplots(3,1,figsize=(12,10), sharey= True) 
    for i in range(3):
        
        plt.suptitle(f"Strain data for three observatories from sample: {sample_id} | Target: {target[a]}")
        sns.lineplot(data=data[i], ax=ax[i], color=sns.color_palette()[i])
        ax[i].legend([labels[i]])
        ax[i].set_xlim(0, 4096)
        ax[i].set_xticks(ticks=[0, 2048, 4096])
        ax[i].set_xticklabels(labels=[0, 1, 2])
   

# function to plot the amplitude spectral density (ASD) plot
def plot_asd(path,
             df,
             target,
             signal_length,
             sample_rate,
             labels = ('LIGO Hanford', 'LIGO Livingston', 'Virgo')
):
    sample_id = df[df['target'] == target].sample(random_state=42)['id'].values[0]
    sample_id = int(sample_id)
    training_files = glob(path)
    data = np.load(training_files[sample_id])

    for i in range(data.shape[0]):
        
        ts = TimeSeries(data[i], sample_rate=sample_rate)
        ax = ts.asd(signal_length).plot(figsize=(12, 5)).gca()
        ax.set_xlim(10, 1024);
        ax.set_title(f"ASD plots for sample: {sample_id} from {labels[i]}")
        
# function to plot the amplitude spectral density (ASD) plot
def plot_asd(sample_id,sample_rate,signal_length):
    # Get the data
    sample = get_data(sample_id)
    
    # we convert the data to gwpy's TimeSeries for analysis
    for i in range(sample.shape[0]):
        ts = TimeSeries(sample[i], sample_rate=sample_rate)
        ax = ts.asd(signal_length).plot(figsize=(12, 5)).gca()
        ax.set_xlim(10, 1024);
        ax.set_title(f"ASD plots for sample: {sample_id} from {obs_list[i]}");
        
def plot_asd_mix(path,
                 df,
                 target,
                 sample_rate,
                 NFFT,
                 f_min,
                 f_max,
                 labels = ('LIGO Hanford', 'LIGO Livingston', 'Virgo')):
    
    sample_id = df[df['target'] == target].sample(random_state=42)['id'].values[0]
    sample_id = int(sample_id)
    training_files = glob(path)
    sample = np.load(training_files[sample_id])
    
   
    Pxx_1, freqs = mlab.psd(sample[0], Fs = sample_rate, NFFT = NFFT)
    Pxx_2, freqs = mlab.psd(sample[1], Fs = sample_rate, NFFT = NFFT)
    Pxx_3, freqs = mlab.psd(sample[2], Fs = sample_rate, NFFT = NFFT)

    psd_1 = interp1d(freqs, Pxx_1)
    psd_2 = interp1d(freqs, Pxx_2)
    psd_3 = interp1d(freqs, Pxx_3)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    ax.loglog(freqs, np.sqrt(Pxx_1),"g",label=labels[0])
    ax.loglog(freqs, np.sqrt(Pxx_2),"r",label=labels[1])
    ax.loglog(freqs, np.sqrt(Pxx_3),"b",label=labels[2])

    ax.set_xlim([f_min, f_max])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Hz^-1/2")
    ax.set_title(f"ASD plots for sample: {sample_id}");
    ax.legend()

    plt.show()

# function to plot the Q-transform spectrogram
def plot_q_transform(path,
                     df
                    ):
    # Get the data
    sample_id = df[df['target'] == target].sample(random_state=42)['id'].values[0]
    sample_id = int(sample_id)
    training_files = glob(path)
    sample = np.load(training_files[sample_id])
    
    # we convert the data to gwpy's TimeSeries for analysis
    for i in range(sample.shape[0]):
        ts = TimeSeries(sample[i], sample_rate=sample_rate)
        ax = ts.q_transform(whiten=True).plot().gca()
        ax.set_xlabel('')
        ax.set_title(f"Spectrogram plots for sample: {sample_id} from {obs_list[i]}")
        ax.grid(False)
        
        ax.set_yscale('log')

# function to plot the Q-transform spectrogram side-by-side
def plot_q_transform_sbs(path,
                         df,
                        sample_rate
                        ):
    # Get the data
    sample_1 = df[df['target'] == 1].sample(random_state=42)['id'].values[0]
    sample_1 = int(sample_1)
    sample_0 = df[df['target'] == 0].sample(random_state=42)['id'].values[0]
    sample_1 = int(sample_0)
    training_files = glob(path)
    sample_1 = np.load(training_files[sample_1])
    sample_0 = np.load(training_files[sample_0])
    
    for i in range(len(obs_list)):
        # get the timeseries
        ts_gw = TimeSeries(sample_1[i], sample_rate=sample_rate)
        ts_no_gw = TimeSeries(sample_0[i], sample_rate=sample_rate)
        
        # get the Q-transform
        image_gw = ts_gw.q_transform(whiten=True)
        image_no_gw = ts_no_gw.q_transform(whiten=True)

        plt.figure(figsize=(20, 10))
        plt.subplot(131)
        plt.imshow(image_gw)
        plt.title(f"id: {sample_1} | Target=1")
        plt.grid(False)

        plt.subplot(132)
        plt.imshow(image_no_gw)
        plt.title(f"id: {sample_0} | Target=0")
        plt.grid(False)
        
        plt.show()
        
def visualize_sample_spectogram(path,
    df, 
    target,
    signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo")
):

    # Get the data
    sample_id = df[df['target'] == target].sample(random_state=42)['id'].values[0]
    sample_id = int(sample_id)
    training_files = glob(path)
    sample = np.load(training_files[sample_id])
    plt.figure(figsize=(16, 5))
    
    for i in range(3):
        X = librosa.stft(sample[i] / sample[i].max())
        Xdb = librosa.amplitude_to_db(abs(sample))
        plt.subplot(1, 3, i + 1)
        librosa.display.specshow(Xdb, sr=2048, x_axis="time", y_axis="hz", vmin=-30, vmax=50) 
        plt.colorbar()
        plt.title(signal_names[i], fontsize=14)

    plt.suptitle(f"Spectrogram plots for sample: {sample_id}", fontsize=16)
    plt.show()
    
def visualize_sample_mfcc(path,
    df, 
    sr=2048,
    signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo")
):
    sample_id = df[df['target'] == target].sample(random_state=42)['id'].values[0]
    sample_id = int(sample_id)
    training_files = glob(path)
    sample = np.load(training_files[sample_id])
    
    plt.figure(figsize=(16, 5))
    for i in range(3):
        mfccs = librosa.feature.mfcc(sample[i] / sample[i].max(), sr=sr)
        plt.subplot(1, 3, i + 1)
        librosa.display.specshow(mfccs, sr=sr, x_axis="time", vmin=-200, vmax=50, cmap="coolwarm")
        plt.title(signal_names[i], fontsize=14)
        plt.colorbar()

    plt.suptitle(f"Mel Frequency Cepstral Coefficients plots for sample: {sample_id}", fontsize=16)
    plt.show()
