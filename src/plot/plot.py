import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d  # interpolating a 1-D function
import matplotlib.mlab as mlab  # some MATLAB commands
from src.load.get_data import *

def visualize_sample(
    sample_id, 
    signal_names=("LIGO Hanford", "LIGO Livingston", "Virgo")
):
    target = train[train['id'] == sample_id]['target'].values
    sample = get_data(sample_id)
    plt.suptitle(f"Strain data for three observatories from sample: {sample_id} | Target:         {target[0]}")
    for i in range(3):
        sns.lineplot(data=sample[i], color=sns.color_palette()[i])
        plt.subplot(4, 1, i + 1)
        plt.plot(sample[i])
        plt.legend([signal_names[i]], fontsize=12, loc="lower right")
        plt.subplot(4, 1, 4)
        plt.plot(sample[i])
    
    plt.subplot(4, 1, 4)
    plt.legend(signal_names, fontsize=12, loc="lower right")
    plt.suptitle(f"Strain data for three observatories from sample: {sample_id} | Target: {target[0]}")
    plt.show()

    
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
        
def plot_asd_mix(sample, sample_rate, NFFT, f_min, f_max):

    Pxx_1, freqs = mlab.psd(sample[0], Fs = sample_rate, NFFT = NFFT)
    Pxx_2, freqs = mlab.psd(sample[1], Fs = sample_rate, NFFT = NFFT)
    Pxx_3, freqs = mlab.psd(sample[2], Fs = sample_rate, NFFT = NFFT)

    psd_1 = interp1d(freqs, Pxx_1)
    psd_2 = interp1d(freqs, Pxx_2)
    psd_3 = interp1d(freqs, Pxx_3)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    ax.loglog(freqs, np.sqrt(Pxx_1),"g",label="Detector 1")
    ax.loglog(freqs, np.sqrt(Pxx_2),"r",label="Detector 2")
    ax.loglog(freqs, np.sqrt(Pxx_3),"b",label="Detector 3")

    ax.set_xlim([f_min, f_max])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Hz^-1/2")
    ax.set_title(f"ASD plots for sample: {sample_id}");
    ax.legend()

    plt.show()
