import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speech Commands, an audio dataset of spoken words designed to help train and evaluate keyword spotting systems. """


_CITATION = """
@article{speechcommandsv2,
   author = { {Warden}, P.},
    title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.03209},
  primaryClass = "cs.CL",
  keywords = {Computer Science - Computation and Language, Computer Science - Human-Computer Interaction},
    year = 2018,
    month = apr,
    url = {https://arxiv.org/abs/1804.03209},
}
"""

_DESCRIPTION = """
This is a set of one-second .wav audio files, each containing a single spoken
English word or background noise. These words are from a small set of commands, and are spoken by a
variety of different speakers. This data set is designed to help train simple
machine learning models. This dataset is covered in more detail at
[https://arxiv.org/abs/1804.03209](https://arxiv.org/abs/1804.03209).
Version 0.01 of the data set (configuration `"v0.01"`) was released on August 3rd 2017 and contains
64,727 audio files.
In version 0.01 thirty different words were recoded: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
"Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow".
In version 0.02 more words were added: "Backward", "Forward", "Follow", "Learn", "Visual".
In both versions, ten of them are used as commands by convention: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go". Other words are considered to be auxiliary (in current implementation
it is marked by `True` value of `"is_unknown"` feature). Their function is to teach a model to distinguish core words
from unrecognized ones.
The `_silence_` class contains a set of longer audio clips that are either recordings or
a mathematical simulation of noise.
"""

_LICENSE = "Creative Commons BY 4.0 License"

_URL = "https://www.tensorflow.org/datasets/catalog/speech_commands"

_DL_URL = "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/{name}/{name}_{split}.tar.gz"

WORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

UNKNOWN_WORDS_V1 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
]

UNKNOWN_WORDS_V2 = UNKNOWN_WORDS_V1 + [
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]

SILENCE = "_silence_"  # background noise
LABELS_V1 = WORDS + UNKNOWN_WORDS_V1 + [SILENCE]
LABELS_V2 = WORDS + UNKNOWN_WORDS_V2 + [SILENCE]


def mel_spectrograms(sr, audio):
    # 40 ms window
    n_fft = int(40.0*sr/1000.0)
    specs = []
    for a in audio:
        mel_spec = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=n_fft, hop_length=n_fft//8,n_mels=80,fmin=10, fmax=6000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db[:,:200]
        x = -80.0*np.ones((80,200))
        x[:,:mel_spec_db.shape[1]] = mel_spec_db
        specs.append(x+80.0)
    return specs


def generate_examples(split):
    basename = os.path.expanduser("~/data/rawSC/v0.02/") + split
    fname = basename + "/index.txt"
    idx = open(fname)
    
    audio = []
    labels = []
    unknown = []
    speaker = []
    utterance = []
    for path in idx:    
        _, word, audio_filename = path.split("/")
        is_unknown = False
        
        if word == SILENCE:
            speaker_id, utterance_id = None, 0
        else:  # word is either in WORDS or unknown
            if word not in WORDS:
                is_unknown = True
            # an audio filename looks like `0bac8a71_nohash_0.wav`
            speaker_id, _, utterance_id = audio_filename.split(".wav")[0].split("_")

            fname = basename+"/"+path.split("./")[1].strip("\n")
            y, sr = librosa.load(fname)
            audio.append(y)
            labels.append(word)
            unknown.append(is_unknown)
            speaker.append(speaker_id)
            utterance.append(utterance_id)
    return sr, audio, labels, unknown, speaker, utterance

def remap(lbl, all_lbl):
    the_lbl = np.unique(all_lbl)
    remap = { x: i for i,x in enumerate(the_lbl)}
    new_lbl = np.vectorize(remap.get)(lbl)
    return new_lbl

if __name__ == '__main__':
    basename = os.path.expanduser("~/data/rawSC/")
    all_labels = []
    all_speakers = []
    labels = {}
    speakers = {}
    for split in ['train', 'validation', 'test' ]:
        sr, audio, labels[split], unknown, speakers[split], utterance = generate_examples(split)
        all_labels.extend(labels[split])
        all_speakers.extend(speakers[split])
        specs = mel_spectrograms(sr, audio)
        np.save(basename+"unknown_"+split, np.asarray(unknown))
        np.save(basename+"utterance_"+split, np.asarray(utterance))
        np.save(basename+"audio_"+split, np.asarray(specs))

    for split in ['train', 'validation', 'test' ]:
        lbl = remap(labels[split],all_labels)
        np.save(basename+"labels_"+split, np.asarray(lbl))
        spk = remap(speakers[split],all_speakers)
        np.save(basename+"speaker_"+split, np.asarray(spk))
