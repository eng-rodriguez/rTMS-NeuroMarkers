"""
visualizations.py
--------

Author: Josue Rodriguez
Date: January 2024
"""

import mne
import matplotlib.pyplot as plt

mne.utils.set_log_level("WARNING")


def plot_eegdata(mne_data, subject, stage, duration=5):
    """
    Plot EEG data.

    This function plots the EEG data using the MNE library. It can handle both continuous raw data
    and epoched data.

    Parameters
    ----------
    mne_data : mne.io.BaseRaw or mne.epochs.BaseEpochs
        The MNE data structure containing the EEG data.

    subject : str
        The identifier of the subject.

    stage : str
        The stage of the EEG data, either "Original" or "Filtered".

    duration : int or None, optional
        The duration of the plot in seconds. Only applicable for mne.io.BaseRaw data.

    Returns
    -------
    None
        This function does not return anything.
    """

    if isinstance(mne_data, mne.io.BaseRaw):
        title = f"{subject.capitalize()} {stage} Continuous EEG Recordings"
        mne_data.plot(
            scalings="auto",
            title=title,
            theme="light",
            show_scalebars=False,
            duration=duration,
        )

    elif isinstance(mne_data, mne.epochs.BaseEpochs):
        title = f"{subject.capitalize()} {stage} Epoched EEG Recordings"
        mne_data.plot(scalings="auto", title=title, theme="light", show_scalebars=False)


def plot_psd(mne_data, subject, stage):
    """
    Plot the power spectral density of the EEG data.

    This function plots the power spectral density of the EEG data using the MNE library.

    Parameters
    ----------
    mne_data : mne.io.BaseRaw or mne.epochs.BaseEpochs
        The MNE data structure containing the EEG data.

    title : str
        The title of the plot.

    Returns
    -------
    None
        This function does not return anything.
    """
    plt.style.use("seaborn-v0_8-paper")
    _, ax = plt.subplots(1, figsize=(10, 5), dpi=300)
    mne_data.plot_psd(ax=ax, fmin=0, fmax=100, show=False)
    ax.set_title(f"{subject.capitalize()} {stage} EEG Recordings PSD")
