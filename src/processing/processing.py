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

""" First, we define the constructor to initialize the configuration of the generator.
Note that here, we assume the path to the data is in a dataframe column.

"""

class DataGenerator(Sequence):

    # For this dataset the list_IDs are the value of the ids
    # for each of the time-series file
    # i.e. for Train data => values of column 'id' from training_labels.csv

    # Also Note we have earlier defined our labels to be the below
    # labels = pd.read_csv(root_dir + "training_labels.csv")
    # and the argument "data" is that label here.
    def __init__(self, path, list_IDs, data, batch_size):
        self.path = path
        self.list_IDs = list_IDs
        self.data = data
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.list_IDs))

    """ __len__ essentially returns the number of steps in an epoch, using the samples and the batch size.
        Each call requests a batch index between 0 and the total number of batches, where the latter is specified in the __len__ method.
        A common practice is to set this value to (samples / batch size)
        so that the model sees the training samples at most once per epoch.
        Now, when the batch corresponding to a given index is called, the generator executes the __getitem__ method to generate it.
    """

    def __len__(self):
        len_ = int(len(self.list_IDs)/self.batch_size)
        if len_ * self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_

    """  __getitem__ method is called with the batch number as an argument to obtain a given batch of data.

    """
    def __getitem__(self, index):
        # get the range to to feed to keras for each epoch
        # incrementing by +1 the bath_size
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    """ And finally the core method which will actually produce batches of data. This private method __data_generation """

    def __data_generation(self, list_IDs_temp):
        # We have 5,60,000 files, each with dimension of 3 * 4096
        X = np.zeros((self.batch_size, 3, 4096))
        y = np.zeros((self.batch_size, 1))
        for i, ID in enumerate(list_IDs_temp):
            id_ = self.data.loc[ID, "id"]
            file = id_ + ".npy"  # build the file name
            path_in = "/".join([self.path, id_[0], id_[1], id_[2]]) + "/"
            # there are three nesting labels inside train/ or test/
            data_array = np.load(path_in + file)
            data_array = (data_array - data_array.mean())/data_array.std()
            X[i, ] = data_array
            y[i, ] = self.data.loc[ID, 'target']
        # print(X)
        return X, y