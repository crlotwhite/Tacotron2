import csv
import librosa
import numpy as np
import pathlib
import random
import re
import tqdm
import unicodedata

from bases.base_preprocessor import BasePreprocessor

class LJSpeechPreprocessor(BasePreprocessor):
    vocab = ' abcdefghijklmnopqrstuvwxyz\'.?'
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}

    def __init__(self, data_path):
        self._root = pathlib.Path(data_path)
        self.metadata = self.root / 'metadata.csv'
        self.wav_files = self.root / 'wavs'

    @property
    def root(self):
        return self._root
    
    def get_meta_data(self):
        meta_data = {
            'sr': 0,
            'train': {
                'audio': [], 
                'text': []
            },
            'test': {
                'audio': [], 
                'text': []
            }
        } 
        
        with open(self.metadata) as file:
            reader = csv.reader(file, delimiter='|')
            lines = list(reader)
        
        random.shuffle(lines)
        
        for i, line in tqdm.tqdm(enumerate(lines)):
            if i < len(lines) * 0.8:
                mode = 'train'
            else:
                mode = 'test'
            meta_data[mode]['text'].append(line[-1])
            y, sr = librosa.load(self.wav_files / f'{line[0]}.wav')
            meta_data['sr'] = sr
            meta_data[mode]['audio'].append(y)
            
        return meta_data
    
    def audio_processing(self, audio, sr):
        mels = librosa.feature.melspectrogram(y=audio, 
                                              sr=sr, 
                                              n_mels=80, 
                                              n_fft=1024, 
                                              hop_length=256, 
                                              win_length=1024, 
                                              fmin=0.0, 
                                              fmax=8000.0)
        return mels

    @classmethod
    def text_processing(cls, text):
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

        text = text.lower()
        text = re.sub('[^{}]'.format(cls.vocab), ' ', text)
        text = re.sub('[ ]+', ' ', text)
        text = [cls.char2idx[char] for char in text]
        text = np.array(text)
        return text