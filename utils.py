from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def train_load(path, parts):
    """
    Example Usage: train_load("/path/to/training/files/", range(1,4)) to load npz files 1-3 inclusive.
    Stacks all the utterances and re-numbers the speaker IDs into a dense integer domain.
    NOTE: Expects the npz files to have already been preprocessed (i.e., in terms of VAD and normalization) using
          the preprocess.py script
    """

    features = []
    speakers = []

    for p in tqdm(parts):
        npz = np.load(os.path.join(path, str(p) + ".preprocessed.npz"), encoding='latin1')

        features.append(npz['feats'])
        speakers.append(npz['targets'])

    features = np.concatenate(features)
    speakers = np.concatenate(speakers)

    nspeakers = densify_speaker_IDs(speakers)

    print("\nLoaded", len(features), "utterances from", nspeakers, "unique speakers.")
    return features, speakers, nspeakers


def dev_load(path):
    """
    Given path to the dev.preprocessed.npz file, loads and returns:
    (1) Dev trials list, where each item is [enrollment_utterance_idx, test_utterance_idx]
    (2) Dev trials labels, where each item is True if same speaker (and False otherwise)
    (3) (Dev) Enrollment array of utterances
    (4) (Dev) Test array of utterances
    """

    assert '.preprocessed.npz' in path

    data = np.load(path, encoding='latin1')
    enrol, test = data['enrol'], data['test']

    print("Loaded dev data.")

    return data['trials'], data['labels'], enrol, test


def test_load(path):
    """
    Given path to the test.preprocessed.npz file, loads and returns:
    (1) Test trials list, where each item is [enrollment_utterance_idx, test_utterance_idx]
    (2) (Test) Enrollment array of utterances
    (3) (Test) Test array of utterances
    """

    assert '.preprocessed.npz' in path

    data = np.load(path, encoding='latin1')
    enrol, test = data['enrol'], data['test']

    print("Loaded test data.")

    return data['trials'], enrol, test


def EER(labels, scores):
    """
    Computes EER (and threshold at which EER occurs) given a list of (gold standard) True/False labels
    and the estimated similarity scores by the verification system (larger values indicates more similar)
    Sources: https://yangcha.github.io/EER-ROC/ & https://stackoverflow.com/a/49555212/1493011
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=True)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def densify_speaker_IDs(speakers):
    """
    Given an array of (integer) speaker IDs, re-numbers the IDs in-place into a dense integer domain,
    from 0 to (# of speakers)-1, and returns # of speakers.
    """

    assert(speakers.dtype in [np.int32, np.int64])

    speaker2ID = {}
    for idx, speaker in enumerate(speakers):
        speaker2ID[speaker] = speaker2ID.get(speaker, len(speaker2ID))
        speakers[idx] = speaker2ID[speaker]

    nspeakers = len(speaker2ID)

    return nspeakers
