# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import mne
import numpy as np
import os
import pickle
from ipdb import set_trace as bp
from tqdm import tqdm

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""

WITH_SPEC = False
STAGE_ONE = False ## produces the processed files for the model training
STAGE_TWO = True ## moves the files once produced into the structure that is used for model training
drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

BAD_FILES = ['028_ses-bckg_028_a__preprocessed-eeg', '093_ses-bckg_093_a_2_preprocessed-eeg', '040_ses-bckg_040_a__preprocessed-eeg', '031_ses-bckg_031_a__preprocessed-eeg', '006_ses-pled_006_a__preprocessed-eeg']

def BuildEvents(signals, times, EventData, spec_true=None, spec_recon=None):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape
    # for i in range(numChan):  # standardize each channel
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    if spec_true is not None:
        spec_recon = np.concatenate([spec_recon, spec_recon, spec_recon], axis=-1)
        spec_true = np.concatenate([spec_true, spec_true, spec_true], axis=-1)
        features_spec_true = np.zeros([numEvents, spec_true.shape[0], spec_true.shape[1], 5])
        features_spec_recon = np.zeros([numEvents, spec_recon.shape[0], spec_recon.shape[1], 5])

    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        if spec_true is not None:
            features_spec_true[i, :, :, :] = spec_true[:, :, int((offset + start) // fs) - 2 : int((offset + end) // fs) + 2]
            features_spec_recon[i, :, :, :] = spec_recon[:, :, int((offset + start) // fs) - 2 : int((offset + end) // fs) + 2]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    if spec_true is not None:
        return [features, offending_channel, labels, features_spec_true, features_spec_recon]
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]

def find_spec(fileName, val=False):
    edf_file = fileName.split('/')[-1]
    
    if val:
        patient_id = fileName.split('/')[-2]
        session_id = edf_file.strip('.edf')
    else:
        patient_id = edf_file.split('_')[0]
        session_id = edf_file.split('_')[1].split('.')[0]
    # recon_dir = "/data/netmit/sleep_lab/EEG_FM/data_EEG/downstream/TUEV/reconstructions_2025-12-15T01-15-13_harvard_vqgan_2_embed32n8192corr01vqtorchema_patchgan_multitaper_128x128_8x16"
    recon_dir = "/data/netmit/sleep_lab/EEG_FM/data_EEG/downstream/TUEV/reconstructions_2025-12-22T22-12-45_harvard_vqgan_2_embed32n8192corr01vqtorchema_patchgan_multitaper_128x128_8x16"
    if val:
        peng_file_name = f"{patient_id}_ses-{session_id}_preprocessed-eeg.npz"
    else:
        peng_file_name = f"{patient_id}_ses-{patient_id}_{session_id}_preprocessed-eeg.npz" # aaaaabji_ses-aaaaabji_00000001_preprocessed-eeg.npz
    if peng_file_name.replace('.npz', '') in BAD_FILES:
        return None, None
    peng_file_path = os.path.join(recon_dir, peng_file_name)
    peng_data = np.load(peng_file_path)
    spec_true = peng_data['original']
    spec_recon = peng_data['reconstruction']
    return spec_true, spec_recon
def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    #signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue

                signals, offending_channels, labels = BuildEvents(signals, times, event)
                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels

def load_up_objects_with_spec(BaseDir, Features, OffendingChannels, Labels, OutDir, val=False):
    import h5py
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    #signals = convert_signals(signals, Rawdata)
                    spec_true, spec_recon = find_spec(dirName + "/" + fname, val=val)
                    if spec_true is None or spec_recon is None:
                        print('fname', dirName + "/" + fname, 'is bad')
                        continue
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue

                # pid = fname.split('_')[0]
                # session = fname.split('_')[1].split('.')[0]
                # other_root_dir = f'/data/netmit/sleep_lab/EEG_FM/data_EEG/downstream/TUEV/{pid}_ses-{pid}_{session}_preprocessed-eeg.h5'
                # h5file = h5py.File(other_root_dir, 'r')
                # other_signals = h5file['recording']['data'][:]  # (time, channels)
                # print('other signal length',other_signals.shape[0] / 200)
                
                print('original signal length',signals.shape[1] / 200, spec_true.shape, spec_recon.shape)
                if spec_true.shape[-1] != signals.shape[1] // 200:
                    ## add one to the end of the spec_true
                    spec_true = np.concatenate([spec_true, spec_true[:, :, -1:]], axis=-1)
                assert spec_true.shape[-1] == signals.shape[1] // 200
                signals, offending_channels, labels, features_spec_true, features_spec_recon = BuildEvents(signals, times, event, spec_true, spec_recon)
                for idx, (signal, offending_channel, label, features_spec_true, features_spec_recon) in enumerate(
                    zip(signals, offending_channels, labels, features_spec_true, features_spec_recon)
                ):
                    sample = {
                        "signal": signal,
                        "spec_true": features_spec_true,
                        "spec_recon": features_spec_recon,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""

root = "/data/netmit/sleep_lab/EEG_FM/TUEV/data/v2.0.1/edf"
if WITH_SPEC:
    train_out_dir = os.path.join(root, "processed_train_with_spec")
    eval_out_dir = os.path.join(root, "processed_eval_with_spec")
else:
    train_out_dir = os.path.join(root, "processed_train")
    eval_out_dir = os.path.join(root, "processed_eval")

if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)


if STAGE_ONE:
    BaseDirTrain = os.path.join(root, "train")
    fs = 200
    TrainFeatures = np.empty(
        (0, 23, fs)
    )  # 0 for lack of intialization, 22 for channels, fs for num of points
    TrainLabels = np.empty([0, 1])
    TrainOffendingChannel = np.empty([0, 1])
    if WITH_SPEC:
        load_up_objects_with_spec(
            BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir, val=False
        )
    else:
        load_up_objects(
            BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
        )

    BaseDirEval = os.path.join(root, "eval")
    fs = 200
    EvalFeatures = np.empty(
        (0, 23, fs)
    )  # 0 for lack of intialization, 22 for channels, fs for num of points
    EvalLabels = np.empty([0, 1])
    EvalOffendingChannel = np.empty([0, 1])
    if WITH_SPEC:
        load_up_objects_with_spec(
            BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir, val=True
        )
    else:
        load_up_objects(
            BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
        )

if STAGE_TWO:
    #transfer to train, eval, and test
    # root = "/share/TUEV/"
    root = "/data/netmit/sleep_lab/EEG_FM/TUEV/data/v2.0.1/edf"
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.2), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    if WITH_SPEC:
        train_folder_name = 'processed_train_with_spec'
        eval_folder_name = 'processed_eval_with_spec'
        test_folder_name = 'processed_test_with_spec'
    else:
        train_folder_name = 'processed_train'
        eval_folder_name = 'processed_eval'
        test_folder_name = 'processed_test'
    for file in tqdm(train_files):
        os.makedirs(os.path.join(root, 'processed', train_folder_name), exist_ok=True)
        os.system(f"mv {os.path.join(root, train_folder_name, file)} {os.path.join(root, 'processed', train_folder_name, file)}")
    for file in tqdm(val_files):
        os.makedirs(os.path.join(root, 'processed', eval_folder_name), exist_ok=True)
        os.system(f"mv {os.path.join(root, train_folder_name, file)} {os.path.join(root, 'processed', eval_folder_name, file)}")
    for file in tqdm(test_files):
        os.makedirs(os.path.join(root, 'processed', test_folder_name), exist_ok=True)
        os.system(f"mv {os.path.join(root, eval_folder_name, file)} {os.path.join(root, 'processed', test_folder_name, file)}")



    os.rmdir(os.path.join(root, train_folder_name))
    os.rmdir(os.path.join(root, eval_folder_name))
    os.rmdir(os.path.join(root, test_folder_name))