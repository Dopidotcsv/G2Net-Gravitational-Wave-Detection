def plot_qtransform(file_id, target, data):
    """Plotting constant Q-transform data.

    Args:
        file_id: str
            unique id of the selected file
        target: int
            0 or 1, target value
        data: numpy.ndarray
            numpy array in the shape (3, 4096), where 3 is the number
            of detectors, 4096 is number of data points (each time series
            instance spans over 2 seconds and is sampled at 2048 Hz)
    """

    times, freqs, qplanes = generate_qtransform(data, fs=fs)

    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(12, 8))

    for i in range(3):
        axs[i].pcolormesh(times, freqs, qplanes[i], shading='auto')
        axs[i].set_yscale('log')
        axs[i].set_ylabel('Frequency (Hz)')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_title(f"Detector {i + 1}", loc='left')
        axs[i].grid(False)

    axs[0].xaxis.set_visible(False)
    axs[1].xaxis.set_visible(False)

    fig.suptitle(f"Q transform visualization. ID: {file_id}. Target: {target}.", fontsize=16)
    plt.show()


