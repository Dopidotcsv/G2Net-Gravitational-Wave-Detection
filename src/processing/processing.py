def load_random_file(signal=None):
    """Selecting a random file from the training dataset.

    Args:
        signal: bool
            optional flag defining whether to select pure detector
            noise (False) or detector noise plus simulated signal (True).
            If skipped, the flag is chosen randomly.
    Returns:
        file_id: str
            unique id of the selected file
        target: int
            0 or 1, target value
        data: numpy.ndarray
            numpy array in the shape (3, 4096), where 3 is the number
            of detectors, 4096 is number of data points (each time series
            instance spans over 2 seconds and is sampled at 2048 Hz)

    """
    if signal is None:
        signal = random.choice([True, False])

    filtered = train_data["target"] == signal  # filtering dataframe based on the target value

    index = random.choice(train_data[filtered].index)  # random index

    file_id = train_data['id'].at[index]
    target = train_data['target'].at[index]
    path = train_data['path'].at[index]

    data = np.load(path)

    return file_id, target, data





def generate_qtransform(data, fs):
    """Function for generating constant Q-transform.

    Args:
        data: numpy.ndarray
            numpy array in the shape (3, 4096), where 3 is the number
            of detectors, 4096 is number of data points (each time series
            instance spans over 2 seconds and is sampled at 2048 Hz)
        fs: int
            sampling frequency
    Returns:
        times: numpy.ndarray
            array of time bins
        freqs: numpy.ndarray
            array of frequency bins
        qplanes: list
            list with 3 elements corresponding to each detector in the raw
            data file. Each element is a 2-d vector of the power in each
            time-frequency bin
    """

    qplanes = []
    for i in range(len(data)):
        # converting data into PyCBC Time Series format
        ts = pycbc.types.TimeSeries(data[i, :], epoch=0, delta_t=1.0 / fs)

        # whitening the data within some frequency range
        ts = ts.whiten(0.125, 0.125)

        # calculating CQT values
        times, freqs, qplane = ts.qtransform(.002, logfsteps=100, qrange=(10, 10), frange=(20, 512))

        qplanes.append(qplane)

    return times, freqs, qplanes
