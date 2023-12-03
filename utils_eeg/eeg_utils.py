import mne
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

STANDARD_10_20 = {
    # The standard montage for EEG recordings has a total of 21 electrodes, including the ground electrode.
    "Fp1": (0.10, 0.67),
    "Fp2": (0.90, 0.67),
    "F7": (0.20, 0.45),
    "F3": (0.30, 0.45),
    "Fz": (0.50, 0.45),
    "F4": (0.70, 0.45),
    "F8": (0.80, 0.45),
    "A1": (0.00, 0.33),
    "T3": (0.25, 0.33),
    "C3": (0.40, 0.33),
    "Cz": (0.50, 0.33),
    "C4": (0.60, 0.33),
    "T4": (0.75, 0.33),
    "A2": (1.00, 0.33),
    "T5": (0.20, 0.20),
    "P3": (0.30, 0.20),
    "Pz": (0.50, 0.20),
    "P4": (0.70, 0.20),
    "T6": (0.80, 0.20),
    "O1": (0.10, 0.00),
    "O2": (0.90, 0.00),
    # 'T1': (0.15, 0.33), # T1 and T2 are not part of the 10-20 system, but are used in the TUH EEG dataset
    # 'T2': (0.85, 0.33),
}


def change_channel_names_TUH(
    raw: mne.io.Raw, montage: str = "standard_1020", apply_montage: bool = False
):
    """
    Function Name: change_channel_names_TUH

    Purpose:

    To rename the channels of a given raw MNE EEG data object to match the standard 10-20 system or other specified montages, with an option to apply the specified montage to the raw EEG object.

    Arguments:

        raw: mne.io.RawArray
        The raw MNE EEG data object whose channels are to be renamed.

        montage: str (default: "standard_1020")
        The name of the montage that the channel names should be matched to. By default, it uses the standard 10-20 system.

        apply_montage: bool (default: False)
        If True, the function will apply the specified montage to the raw EEG object. If False, only channel names will be changed without applying any montage.

    Returns:

        mne.io.RawArray
        The updated raw MNE EEG data object with renamed channels and optionally with the montage applied.

    Notes:

        The function uses the STANDARD_10_20 dictionary, which maps standard channel names from the 10-20 system to their relative positions. This dictionary also includes the T1 and T2 channels, which are not part of the standard 10-20 system but are commonly used in the TUH EEG dataset.

        The function searches for each standard channel name in the raw EEG object's channel names (ignoring case) and renames them to match the standard naming convention.

        Any prefix "EEG " and suffix "-REF" are also removed from the channel names, which is specific to the naming convention in the TUH EEG dataset.

        If apply_montage is set to True, the function will apply the specified montage to the raw EEG object using the mne.channels.make_standard_montage and raw.set_montage functions.

    Usage Example:

        raw = mne.io.read_raw_eeglab('path_to_raw_eeg_data.set')
        updated_raw = change_channel_names_TUH(raw, montage="standard_1020", apply_montage=True)

    Caution:

        Before using this function, ensure that the STANDARD_10_20 dictionary is defined in the same scope or is imported from a utility module, as indicated by the commented out import line in the function (# from utils.eeg_utils import STANDARD_10_20).

    """
    # from utils.eeg_utils import STANDARD_10_20

    channel_names = raw.ch_names.copy()

    for i, channel_name in enumerate(channel_names):
        for standard_channel_name in STANDARD_10_20.keys():
            if standard_channel_name.lower() in channel_name.lower():
                logging.debug(standard_channel_name)
                channel_names[i] = standard_channel_name

        # strip channel name of 'EEG' and -REF (Specific to TUH EEG dataset)
        channel_names[i] = (
            channel_names[i].replace("EEG ", "").replace("-REF", "").strip()
        )

    # print(channel_names)

    # create dictionary of old and new channel names
    channel_names_dict = dict(zip(raw.ch_names, channel_names))
    logging.info(f"Channel names dictionary: {channel_names_dict}")

    # return a copy of raw with new channel names
    raw_new = raw.copy()
    raw_new.rename_channels(channel_names_dict)

    if apply_montage:
        logging.info(f"Applying {montage} montage")
        print(f"Applying {montage} montage")
        montage = mne.channels.make_standard_montage(montage)
        raw_new.set_montage(montage, on_missing="warn")
    else:
        logging.info("Not applying montage")
        print("Not applying montage")

    return raw_new


def set_eventmapping_TUH(
    raw: mne.io.Raw, annotations_path_tuh: str = "", skiprows: int = 5, plot=True
):
    """
    Overview:

    This function reads annotations from a specified file, maps them to unique event IDs, and then updates a given MNE Raw object with these annotations. Optionally, it can also plot the events.
    Parameters:

        raw (mne.io.Raw): An MNE raw object containing EEG data.

        annotations_path_tuh (str): The path to the file containing the TUH annotations. This parameter is required, and if left empty, the function will raise a ValueError.

        skiprows (int): The number of rows to skip at the beginning of the annotation file when reading it. Default value is 5.

        plot (bool): If True, the function will plot the events. Default value is True.

    Returns:

        raw_ (mne.io.Raw): An updated MNE raw object with the annotations added.

        events_array (np.ndarray): A NumPy array containing the annotations. The first column represents the start time, the second column represents the stop time, and the third column represents the event ID.

        event_mapping (dict): A dictionary mapping the unique labels from the annotations to their corresponding event IDs.

    Additional Information:

        The function first checks if the annotations_path_tuh parameter is empty. If it is, a ValueError is raised.

        The TUH annotations are then loaded into a Pandas DataFrame using the pd.read_csv method, skipping the specified number of rows.

        An empty events array is initialized, and then filled with start times, stop times, and event IDs.

        The labels from the annotations are mapped to unique event IDs using a dictionary. This mapping is then used to populate the third column of the events array.

        The MNE raw object is updated with these annotations using the set_annotations method.

        If the plot parameter is True, the events are plotted using MNE's plot_events method.

    Example Usage:

    python

    raw_updated, events, mapping = set_eventmapping_TUH(raw=my_raw_object, annotations_path_tuh="path/to/annotations.csv")


    """
    raw_ = raw.copy()

    if not annotations_path_tuh:
        raise ValueError("annotations_path_tuh is empty")

    tuh_annotations = pd.read_csv(annotations_path_tuh, skiprows=skiprows)
    logging.info(f"Loaded annotations from {annotations_path_tuh}")
    logging.info(f"Number of annotations: {len(tuh_annotations)}")

    # create events_array
    events_array = np.zeros((len(tuh_annotations), 3), dtype=np.float32)
    events_array[:, 0] = tuh_annotations["start_time"].values
    events_array[:, 1] = tuh_annotations["stop_time"].values

    # convert labels to IDs and add to events_array
    labels = tuh_annotations["label"].values
    logging.debug(labels)
    labels_unique = np.unique(labels)
    print(f"Unique labels: {labels_unique}")
    label_to_id = {label: i + 1 for i, label in enumerate(labels_unique)}
    events_array[:, 2] = [label_to_id[label] for label in labels]

    # add events_array to raw object
    # update the events in the raw object
    raw_.set_annotations(
        mne.Annotations(
            onset=events_array[:, 0],
            duration=events_array[:, 1] - events_array[:, 0],
            description=events_array[:, 2],
            orig_time=None,
        )
    )

    if plot:
        events = mne.events_from_annotations(raw_)[0]
        logging.debug(events)
        logging.debug(label_to_id)
        fig, ax = plt.subplots(figsize=[15, 5])

        mne.viz.plot_events(events, raw_.info["sfreq"], event_id=label_to_id, axes=ax)
        plt.show()

    event_mapping = label_to_id

    return raw_, events_array, event_mapping
