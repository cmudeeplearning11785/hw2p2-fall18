from __future__ import print_function
import sys
import numpy as np

# Set these appropriate
path_prefix = sys.argv[1].strip()
part_idx = sys.argv[2].strip()

# VAD Parameters (also, see basic normalization below)
VAD_THRESHOLD = -80   # if a frame has no filter that exceeds this threshold, it is assumed silent and removed
VAD_MIN_NFRAMES = 150  # if a filtered utterance is shorter than this after VAD, the full utterance is retained


assert(path_prefix)
assert(path_prefix)
assert(part_idx in map(lambda x: str(x), list(range(1, 7))) or part_idx in ["dev", "test"])
assert(VAD_THRESHOLD >= -100.0)
assert(VAD_MIN_NFRAMES >= 1)


def bulk_VAD(feats):
    return [normalize(VAD(utt)) for utt in feats]


def VAD(utterance):
    filtered = utterance[utterance.max(axis=1) > VAD_THRESHOLD]
    return utterance if len(filtered) < VAD_MIN_NFRAMES else filtered


def normalize(utterance):
    utterance = utterance - np.mean(utterance, axis=0, dtype=np.float64)
    return np.float16(utterance)


npz = np.load(path_prefix + str(part_idx) + ".npz", encoding='latin1')

if part_idx == "dev":
    enrol, test = bulk_VAD(npz['enrol']), bulk_VAD(npz['test'])
    np.savez(path_prefix + str(part_idx) + ".preprocessed.npz", enrol=enrol, test=test, trials=npz['trials'], labels=npz['labels'])

elif part_idx == "test":
    enrol, test = bulk_VAD(npz['enrol']), bulk_VAD(npz['test'])
    np.savez(path_prefix + str(part_idx) + ".preprocessed.npz", enrol=enrol, test=test, trials=npz['trials'])

else:
    feats, targets = bulk_VAD(npz['feats']), npz['targets']
    np.savez(path_prefix + str(part_idx) + ".preprocessed.npz", feats=feats, targets=targets)
