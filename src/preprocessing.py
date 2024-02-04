"""
preprocessing.py
--------

Author: Josue Rodriguez
Date: January 2024
"""

import mne
import numpy as np

from scipy.signal import detrend

mne.utils.set_log_level("WARNING")


def create_epochs(data, start_times, duration):
    """
    Create epochs from raw EEG data.

    This function segments the raw EEG data into epochs based on
    the provided start times and duration. Each epoch represents a
    specific time window of the EEG recording.

    Parameters
    ----------
    data : mne.io.Raw
        The raw EEG data.

    start_times : list
        A list of start times for each epoch.

    duration : float
        The duration of each epoch in seconds.

    Returns
    -------
    epochs : mne.Epochs
        An Epochs object containing the segmented data.

    Examples
    --------
    >>> data = load_raw_eeg_data()  # Load raw EEG data
    >>> start_times = [0, 5, 10]  # Start times for each epoch (in seconds)
    >>> duration = 2  # Duration of each epoch (in seconds)
    >>> epochs = create_epochs(data, start_times, duration)  # Create epochs
    >>> print(epochs)  # Print the Epochs object

    Notes
    -----
    The start times should be specified in seconds relative to the beginning of the recording.
    The resulting epochs will have a baseline correction applied and will be preloaded into memory.
    """
    sfreq = data.info["sfreq"]

    events = [[int(start_time * sfreq), 0, 1] for start_time in start_times]

    epochs = mne.Epochs(
        data,
        events=events,
        event_id={"start": 1},
        tmin=0,
        tmax=duration - (1 / sfreq),
        baseline=None,
        preload=True,
    )

    return epochs


def notch_filter(data, freqs):
    """
    Apply notch filter to the given data.

    This function applies a notch filter to remove specific frequencies from the data.
    It takes in the raw data and a list of frequencies to be filtered.

    Parameters
    ----------
    data : mne.io.Raw
        The raw data to be filtered.

    freqs : list
        A list of frequencies to be filtered.

    Returns
    -------
    mne.io.Raw
        The notch-filtered data.

    Examples
    --------
    >>> import mne
    >>> raw_data = mne.io.Raw(...)  # create or load the raw data
    >>> frequencies = [50, 60]  # frequencies to be filtered
    >>> filtered_data = notch_filter(raw_data, frequencies)
    """
    return data.copy().notch_filter(freqs)


def bandpass_filter(data, l_freq, h_freq):
    """
    Apply a bandpass filter to the given data.

    This function applies a bandpass filter to the raw data, allowing only
    frequencies within the specified passband to pass through.It takes in the
    raw data, the lower frequency of the passband (l_freq), and the higher
    frequency of the passband (h_freq).

    Parameters
    ----------
    data : mne.io.Raw
        The raw data to be filtered.

    l_freq : float
        The lower frequency of the passband.

    h_freq : float
        The higher frequency of the passband.

    Returns
    -------
    mne.io.Raw
        The bandpass-filtered data.

    Examples
    --------
    >>> import mne
    >>> raw_data = mne.io.Raw(...)  # Replace ... with actual raw data
    >>> filtered_data = bandpass_filter(raw_data, 1.0, 50.0)
    >>> filtered_data.plot()  # Plot the filtered data
    """
    return data.copy().filter(l_freq, h_freq)


def re_reference(data, ref_channels="average"):
    """
    Re-reference the EEG data.

    This function re-references the EEG data by subtracting the average
    reference or by using specific reference channels.It takes in the
    raw data and the reference channels as input.

    Parameters
    ----------
    data : mne.io.Raw
        The raw EEG data.

    ref_channels : str or list of str, optional
        The channels to use as reference. Default is "average" for average reference.
        If a list of channel names is provided, the data will be re-referenced
        using only those channels.

    Returns
    -------
    mne.io.Raw
        The re-referenced EEG data.

    Notes
    -----
    - If ref_channels is set to "average", the average reference will be
        subtracted from all EEG channels.
    - If ref_channels is a list of channel names, the data will be
        re-referenced using only those channels.

    Examples
    --------
    >>> re_reference(raw_data)  # Re-reference using average reference
    >>> re_reference(raw_data, ref_channels=["Cz", "Fz"])  # Re-reference using specific channels
    """
    return data.copy().set_eeg_reference(ref_channels)


def realign_final_session(mne_raw_sessions):
    """
    Realign the final session.

    This function splits the final session into two halves and appends
    the second half to the list of sessions. This is done to ensure that
    the final session is the same length as the other sessions.

    Parameters
    ----------
    mne_data : list
        A list of mne.io.Raw objects containing the EEG data for each session.

    Returns
    -------
    list
        A list of mne.io.Raw objects containing the EEG data for each session.

    Examples
    --------
    >>> mne_data = load_raw_eeg_data()  # Load raw EEG data
    >>> mne_data = realign_final_session(mne_data)  # Realign final session

    Notes
    -----
    This function is only necessary if the final session is longer than the other sessions.
    """
    raw_final_session = mne_raw_sessions[-1]
    midpoint_samples = raw_final_session.n_times // 2

    # Split final session into two halves
    raw_first_half = raw_final_session.copy().crop(
        tmax=midpoint_samples / raw_final_session.info["sfreq"]
    )
    raw_second_half = raw_final_session.copy().crop(
        tmin=midpoint_samples / raw_final_session.info["sfreq"]
    )

    mne_raw_sessions[-1] = raw_first_half
    mne_raw_sessions.append(raw_second_half)

    return mne_raw_sessions


def interpolate_nans(data):
    """
    Interpolate NaN values in the given data.

    This function interpolates NaN values in the given data by
    replacing them with the average of the neighboring values.

    Parameters
    ----------
    data : ndarray
        The data to be interpolated.

    Returns
    -------
    ndarray
        The interpolated data.

    Examples
    --------
    >>> interpolate_nans(data)
    """
    for channel in range(data.shape[0]):
        channel_data = data[channel]
        nans = np.isnan(channel_data)
        indices = np.arange(len(channel_data))
        channel_data[nans] = np.interp(
            indices[nans], indices[~nans], channel_data[~nans]
        )

    return data


def signal_detrend(data):
    """
    Detrend the given data.

    This function detrends the given data by subtracting the mean from each channel.

    Parameters
    ----------
    data : mne.io.Raw
        The raw EEG data.

    Returns
    -------
    mne.io.Raw
        The detrended EEG data.

    Examples
    --------
    >>> detrend(raw_data)  # Detrend the data
    """
    return data.copy().apply_function(detrend, picks="eeg", channel_wise=True)
