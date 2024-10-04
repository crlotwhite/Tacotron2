import abc
import h5py

def key_check(meta_data: dict, key: str):
    assert meta_data.get(key, None) is not None, f'There is no {key} in meta_data'

def meta_data_checker(meta_data: dict):
    assert isinstance(meta_data, dict), 'meta_data type is not dict'
    key_check(meta_data, 'train')
    key_check(meta_data, 'test')
    key_check(meta_data['train'], 'text')
    key_check(meta_data['train'], 'audio')
    key_check(meta_data['test'], 'text')
    key_check(meta_data['test'], 'audio')

class Preprocessor(abc.ABC):
    @property
    @abc.abstractmethod
    def root(self):
        pass

    @abc.abstractmethod
    def get_meta_data(self):
        pass
    
    @abc.abstractmethod
    def text_processing(self, text):
        pass
    
    @abc.abstractmethod
    def audio_processing(self, audio, sr):
        pass

    def process(self):
        meta_data = self.get_meta_data()
        meta_data_checker(meta_data)
        
        with h5py.File(self.root / 'data.hdf5', 'w') as f:
            for mode in ['train', 'test']:
                for i, (audio, text) in enumerate(zip(meta_data[mode]['audio'], meta_data[mode]['text'])):
                    processed_text = self.text_processing(text)
                    processed_audio = self.audio_processing(audio, meta_data['sr'])
                    f.create_dataset(f'{mode}/mel_spectrograms/{i}', data=processed_audio)
                    f.create_dataset(f'{mode}/texts/{i}', data=processed_text)
