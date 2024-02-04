"""
utils.py
--------

Author: Josue Rodriguez
Date: January 2024
"""

import os
import mne
import numpy as np
import pandas as pd

mne.utils.set_log_level("WARNING")


def load_data(directory, filename, columns_to_remove=None):
    """
    Load data from a file in the given directory.

    Parameters
    ----------
    directory : str
        Directory where the file is located.

    filename : str
        Name of the file to load.

    columns_to_remove : list, optional
        List of columns to remove from the data. Default is None.

    Returns
    -------
    data : ndarray
        Data loaded from the file.

    Notes
    -----
    - The data file should be in a tabular format.
    - The function assumes that the data file has a header row.
    - Non-numeric values in the data will be converted to NaN.
    - If `columns_to_remove` is provided, the specified columns will be removed from the data.

    Examples
    --------
    >>> directory = '/path/to/data'
    >>> filename = 'data.csv'
    >>> columns_to_remove = ['column1', 'column2']
    >>> data = load_data(directory, filename, columns_to_remove)
    """
    filepath = os.path.join(directory, filename)
    data = pd.read_table(filepath, header=None, low_memory=False)

    # Convert all columns to numeric, non-numeric values will be NaN
    data = data.apply(pd.to_numeric, errors="coerce")

    if columns_to_remove:
        data = data.drop(columns_to_remove, axis=1)

    return data.values


def load_channels_info(directory, filename):
    """
    Load channels information from a file in the given directory.

    This function reads a file located in the specified directory and extracts the 
    channels information. The file should be in a tabular format, where each row 
    represents a channel and each column contains relevant information about the channel. 
    The last column of the file should contain the channel names.

    Parameters
    ----------
    directory : str
        The directory where the file is located.

    filename : str
        The name of the file to load.

    Returns
    -------
    channels_info : list of tuples
        A list of tuples containing the channels information loaded from the file. 
        Each tuple consists of the channel name and its type.
    """
    filepath = os.path.join(directory, filename)
    channels = pd.read_table(filepath, header=None, low_memory=False)
    ch_names = channels.iloc[:, -1].str.strip().to_list()
    ch_types = ["eeg"] * len(ch_names)

    channels_info = list(zip(ch_names, ch_types))

    return channels_info


def load_mne_data(directory, filename):
    """
    Load data from a file in the given directory using MNE.

    This function reads data from a file located in the specified directory 
    using the MNE library. The file can be in either raw or epochs format, 
    indicated by the file extension. Supported file formats include "_raw.fif" 
    for raw data and "_epo.fif" for epochs data.

    Parameters
    ----------
    directory : str
        The directory where the file is located.

    filename : str
        The name of the file to load.

    Returns
    -------
    data : mne.io.Raw or mne.Epochs
        The data loaded from the file.
    """
    filepath = os.path.join(directory, filename)

    if filename.endswith(("_raw.fif", "_epo.fif")):
        loader_func = (
            mne.io.read_raw_fif if filename.endswith("_raw.fif") else mne.read_epochs
        )
        data = loader_func(filepath, preload=True, verbose=False)
    else:
        raise ValueError("Invalid file format. Supported formats: _raw.fif, _epo.fif")

    return data


def save_mne_data(data, directory, filename):
    """
    Save MNE data object to a file in the specified directory.

    This function saves the MNE data object to a file in the specified directory. 
    The file will be saved with the given filename. If a file with the same name 
    already exists, it will be overwritten.

    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        The MNE data object to save.

    directory : str
        The directory where the file will be saved.

    filename : str
        The name of the file to save.

    Returns
    -------
    None
        This function does not return anything.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    data.save(filepath, overwrite=True)


def create_montage(montage_name):
    """
    Create a montage using MNE.

    This function creates a montage using the MNE library. A montage is a 
    collection of electrode positions that define the spatial layout of 
    electrodes on the scalp. It is commonly used in EEG data analysis
    to map the electrode positions to the corresponding channels.

    Parameters
    ----------
    montage_name : str
        Name of the montage to create. The available montage names can 
        be found in the MNE documentation.

    Returns
    -------
    montage : mne.channels.Montage
        Montage created using MNE. The montage object contains the electrode 
        positions and can be used to apply the montage to EEG data.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    return montage


def create_mne_raw(data, channels_info, montage, sfreq):
    """
    Create a raw object using MNE.

    This function creates a raw object using the MNE library. A raw object 
    represents continuous EEG data and provides various methods for data 
    manipulation and analysis.

    Parameters
    ----------
    data : ndarray
        The data to create the raw object. The shape of the data should be (n_channels, n_samples).

    channels_info : list of tuples
        A list of channel information. Each tuple should contain the channel name and channel type.

    montage : mne.channels.Montage
        The montage to use for the raw object. A montage is a collection of electrode 
        positions that define the spatial layout of electrodes on the scalp.

    sfreq : float
        The sampling frequency of the data.

    Returns
    -------
    raw : mne.io.Raw
        The raw object created using MNE.
    """
    data = np.transpose(data)
    ch_names, ch_types = zip(*channels_info)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)

    return raw
