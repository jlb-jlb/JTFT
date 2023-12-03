import mne
import os
import PIL

import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from skimage.transform import resize
from matplotlib import image
from sklearn.preprocessing import MinMaxScaler
from utils.data_loader import DataLoader
from utils.tuh_eeg_utils import tuh_eeg_load_data_with_annotations
from utils.tuh_eeg_utils import tuh_eeg_apply_TUH_bipolar_montage, tuh_eeg_create_epochs_with_labels

logging.basicConfig(level=logging.INFO 
                    , format='%(asctime)s %(levelname)s %(message)s', force=True)

mne.set_log_level('WARNING')


class config:
    data_loader_path = "/home/joscha.l.bisping/bachelor_thesis/seizure_detection_tutorial/TUH_EEG_SEIZ/edf/train"
    target_path = "/home/joscha.l.bisping/bachelor_thesis/seizure_detection_tutorial/preprocessed_dataset_freq_and_raw/train"
    data_extensions = ['.edf', '.csv_bi']
    l_freq = 0.5  # high-pass filter cutoff (Hz)
    h_freq = 50   # low-pass filter cutoff (Hz)
    epoch_duration = 6  # seconds
    window_overlap = 2  # seconds
    new_sfreq = 50  # Hz
    cutoff_freq_fft = 50  # Hz
    scaler_range = (0, 255)
    resize_width = 224 # pixels
    resize_height = 224 # pixels (will be applied to the raw data and the FFT data)
    line_thickness = 2 # pixels


def main(data_loader=None):

    logging_level = logging.getLogger().level
    logging.error(f"Logging level: {logging_level}")
    logging.info("Loading data...")
    if data_loader is None:
        data_loader = DataLoader(
            config.data_loader_path,
            extensions=config.data_extensions,
        )
    
        logging.info(data_loader.describe())
        files = data_loader.get_next()
    else:
        files = data_loader.get_current_file_tuple()
        logging.info(f"Current index: {data_loader.current_index}")


    
    logging.info(files)

    logging.info(f"Filtering data with l_freq={config.l_freq} and h_freq={config.h_freq}...")
    logging.info(f"Configuration of everything: {config.__dict__}")

    while files is not None:
        # load data
        raw_eeg = tuh_eeg_load_data_with_annotations(files[0], files[1])

        # bipolar montage
        try:
            raw_eeg = tuh_eeg_apply_TUH_bipolar_montage(raw_eeg, files[0], only_return_bipolar=True)
        except ValueError as e:
            logging.error(f"Error while applying bipolar montage: {e}")
            logging.error(f"Froblematic file: {files[0]}")
            logging.error("Skipping this file...")
            files = data_loader.get_next()
            logging.error(f"Skipping to next file: {files[0]}")
            logging.error(f"loop count: {data_loader.current_index}")
            continue #####################################################

        # filter data
        # filtered_eeg = raw_eeg.copy().filter(config.l_freq, config.h_freq, fir_design='firwin', method='fir', verbose=False)

        

        # epochs
        try:    
            epochs, labels = tuh_eeg_create_epochs_with_labels(raw_eeg, window_length=config.epoch_duration, overlap=config.window_overlap, verbose=False)
            # epochs, labels = tuh_eeg_create_epochs_with_labels(raw_eeg, window_length=5, overlap=1, verbose=True)
            logging.info(f"epochs.shape: {len(epochs)}")
        except ValueError as e:
            logging.error(f"Error while creating epochs: {e}")
            logging.error(f"Froblematic file: {files[0]}")
            logging.error("Skipping this file...")
            files = data_loader.get_next()
            logging.error(f"Skipping to next file: {files[0]}")
            logging.error(f"loop count: {data_loader.current_index}")
            continue #####################################################


        # get frequencies
        frequencies = np.fft.rfftfreq(epochs.get_data()[0].shape[1], 1/epochs.info['sfreq'])
        logging.info(f"frequencies.shape: {frequencies.shape}")
        idx = np.where(frequencies == config.cutoff_freq_fft)[0][0]
        logging.info(f"np.where {config.cutoff_freq_fft} Hz: {idx}")

        idx_low_freq = np.where(frequencies == config.l_freq)[0][0]


        # get ffts
        ffts = []
        for sample in epochs:
            spectra = np.abs(np.fft.rfft(sample))
            logging.debug(f"spectra.shape: {spectra.shape}")
            spectra_abs_pow2 = spectra  # ** 2
            logging.debug(f"spectra_abs_pow2.shape: {spectra_abs_pow2.shape}")
            # scaler with feature range from 0 - 255
            scaler_minmax = MinMaxScaler(feature_range=config.scaler_range)
            minmax_scaled_spectra = scaler_minmax.fit_transform(spectra_abs_pow2[:, idx_low_freq:idx])
            # print(minmax_scaled_spectra.shape)
            # print(minmax_scaled_spectra[0:2])
            logging.debug(f"minmax_scaled_spectra.shape: {minmax_scaled_spectra.shape}")

            fft_image = Image.fromarray(minmax_scaled_spectra)
            fft_image = fft_image.resize((config.resize_width, config.resize_height), resample=PIL.Image.LANCZOS)
            ffts.append(np.array(fft_image))


        ffts = np.array(ffts)


        filtered_eeg = raw_eeg.copy().filter(config.l_freq, config.h_freq, fir_design='firwin', method='fir', verbose=False)

        # epochs
        try:    
            epochs, labels = tuh_eeg_create_epochs_with_labels(raw_eeg, window_length=config.epoch_duration, overlap=config.window_overlap, verbose=False)
            # epochs, labels = tuh_eeg_create_epochs_with_labels(raw_eeg, window_length=5, overlap=1, verbose=True)
            logging.info(f"epochs.shape: {len(epochs)}")
        except ValueError as e:
            logging.error(f"Error while creating epochs: {e}")
            logging.error(f"Froblematic file: {files[0]}")
            logging.error("Skipping this file...")
            files = data_loader.get_next()
            logging.error(f"Skipping to next file: {files[0]}")
            logging.error(f"loop count: {data_loader.current_index}")
            continue #####################################################

        # downsample epochs
        epochs = epochs.load_data().resample(config.new_sfreq, npad='auto')
        logging.info(f"epochs.shape: {len(epochs)}")
        logging.info(f"epochs.info['sfreq']: {epochs.info['sfreq']}")

        # scale epochs
        scaled_resized_epochs = []
        for sample in epochs:
            scaler_minmax = MinMaxScaler(feature_range=config.scaler_range)
            minamx_scaled_sample = scaler_minmax.fit_transform(sample)

            raw_image = Image.fromarray(minamx_scaled_sample)
            raw_image = raw_image.resize((config.resize_width, config.resize_height), resample=PIL.Image.LANCZOS)
            scaled_resized_epochs.append(np.array(raw_image))


        scaled_resized_epochs = np.array(scaled_resized_epochs)



        # lists for data
        pictures = []
        metadata = []
        for i, (fft, epoch, label) in enumerate(zip(ffts, scaled_resized_epochs, labels)):
            
            rgb_image = np.zeros((config.resize_width, config.resize_height, 3), dtype=np.uint8)

            # red channel 
            rgb_image[:, :, 0] = epoch

            # green channel
            rgb_image[:, :, 1] = fft

            rgb_image = Image.fromarray(rgb_image)

            # pictures.append(rgb_image)

            filename = "-".join(files[0].split("/")[-4:])  # split path and delete everything that is prior to the last folder of TRAINING_PATH
            filename = filename.replace(".edf", "--") + f"{i}.png".zfill(9)



            metadata.append({
                "file_name": filename,
                "label": label,
                "epoch_index": i,
            })
            logging.debug(metadata[-1])

            # save image
            rgb_image.save(f"{config.target_path}/{filename}")
        
        logging.info(f"len(metadata): {len(metadata)}")
        logging.info(f"len(pictures): {len(pictures)}")
        logging.info(f"len(labels): {len(labels)}")
        logging.info(f"last item in metadata: {metadata[-1]}")


        # # save the pictures as png
        # logging.info(f"saving {len(pictures)} pictures to {config.target_path}")
        # for i, (picture, metad) in enumerate(zip(pictures, metadata)):
        #     image.imsave(f"{config.target_path}/{metad['file_name']}", picture)

        # add metadata to metadata.jsonl file in config.target_path folder
        logging.info(f"saving metadata to {config.target_path}/metadata.jsonl")
        df = pd.DataFrame(metadata)
        if os.path.exists(f"{config.target_path}/metadata.jsonl"):
            df.to_json(f"{config.target_path}/metadata.jsonl", orient="records", lines=True, mode="a")
        else:
            df.to_json(f"{config.target_path}/metadata.jsonl", orient="records", lines=True, mode="w")


        # load the next files!
        logging.info("Loading next files...")
        logging.info(f"Current index: {data_loader.current_index}")
        files = data_loader.get_next()
        logging.info(files)


def check_target_path(target_path, data_loader=None):
    # check if target path exists
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if len(os.listdir(target_path)) != 0 and data_loader is None:
        logging.error(f"Target path {target_path} is not empty!")
        # ask user if he wants to overwrite the files
        # if yes, delete the files
        # if no, exit
        input_user = input("Do you want to overwrite the files? (y/n/keep going [k]): ")
        if input_user == "y":
            logging.info("Deleting files...")
            for file in os.listdir(target_path):
                os.remove(os.path.join(target_path, file))
            logging.info("Files deleted!")
        elif input_user == "k":
            logging.info("Continuing...")
        else:
            logging.info("Exiting...") 
            exit(1)
    elif data_loader is not None:
        logging.info("Not checking target path because data_loader is not None. Assuming that we continue with the next files...")
    else:
        logging.info("Target path is empty. Continuing...")

if __name__ == "__main__":
    
    # change config path
    config.data_loader_path = "/home/Bachelor-Thesis-JLB/TUH_EEG_SEIZ/edf/train"
    config.target_path = "/home/Bachelor-Thesis-JLB/FT_RAW_DATA_COMBO/train"  
        
    
    
    logging.basicConfig(level=logging.INFO 
                    , format='%(asctime)s %(levelname)s %(message)s', filename="./logs_20_train_freq_and_raw.log",force=True)
    logging.info("Starting with training data...")
    logging.info(f"Checking target path {config.target_path}...")
    

    # data_loader = DataLoader(
    #     config.data_loader_path,
    #     extensions=config.data_extensions,
    # )

    # files = data_loader.get_current_file_tuple()

    # while(files[0] != "/home/Bachelor-Thesis-JLB/TUH_EEG_SEIZ/edf/train/aaaaaqtx/s033_2014_04_02/01_tcp_ar/aaaaaqtx_s033_t002.edf"):
    #     print(files[0], end="\r")
    #     files = data_loader.get_next()

    # check_target_path(config.target_path, data_loader=data_loader)

    # main(data_loader=data_loader)

    # check_target_path(config.target_path)
    # main()

    # dev
    config.data_loader_path = "/home/Bachelor-Thesis-JLB/TUH_EEG_SEIZ/edf/dev"
    config.target_path = "/home/Bachelor-Thesis-JLB/FT_RAW_DATA_COMBO/dev"
    logging.basicConfig(level=logging.INFO
                    , format='%(asctime)s %(levelname)s %(message)s', filename="./logs_20_dev_freq_and_raw.log",force=True)
    logging.info("Starting with dev data...")
    logging.info(f"Checking target path {config.target_path}...")
    print("Starting with dev data...")
    print(f"Checking target path {config.target_path}...")
    check_target_path(config.target_path)
    
    main()

    # # eval
    # config.data_loader_path = "/home/joscha.l.bisping/bachelor_thesis/seizure_detection_tutorial/TUH_EEG_SEIZ/edf/eval"
    # config.target_path = "/home/joscha.l.bisping/bachelor_thesis/seizure_detection_tutorial/preprocessed_dataset_freq_and_raw/eval"
    # logging.basicConfig(level=logging.INFO
    #                 , format='%(asctime)s %(levelname)s %(message)s', filename="./logs_eval_freq_and_raw.log",force=True)
    # main()
