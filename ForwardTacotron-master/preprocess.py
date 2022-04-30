import argparse
from multiprocessing import Pool, cpu_count
from random import Random
from typing import Tuple, Dict

import pyworld as pw

from utils.display import *
from utils.dsp import *
from utils.files import get_files, pickle_binary, read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.recipes import ljspeech


# Helper functions for argument types
from utils.text.tokenizer import Tokenizer


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


class Preprocessor:

    def __init__(self,
                 paths: Paths,
                 text_dict: Dict[str, str],
                 cleaner: Cleaner,
                 lang: str,
                 dsp: DSP):
        self.paths = paths
        self.text_dict = text_dict
        self.cleaner = cleaner
        self.lang = lang
        self.dsp = dsp

    def __call__(self, path: Path) -> Tuple[str, int, str]:
        wav_id = path.stem
        m, x, raw_pitch = self._convert_file(path)
        np.save(self.paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
        np.save(self.paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
        np.save(self.paths.raw_pitch/f'{wav_id}.npy', raw_pitch, allow_pickle=False)
        text = self.text_dict[wav_id]
        text = self.cleaner(text)
        return wav_id, m.shape[-1], text

    def _convert_file(self, path: Path) -> Tuple[np.array, np.array, np.array]:
        y = self.dsp.load_wav(path)
        if self.dsp.trim_long_silences:
           y = self.dsp.trim_long_silences(y)
        if self.dsp.should_trim_start_end_silence:
           y = self.dsp.trim_silence(y)
        peak = np.abs(y).max()
        if self.dsp.should_peak_norm or peak > 1.0:
            y /= peak
        mel = self.dsp.wav_to_mel(y)
        pitch, _ = pw.dio(y.astype(np.float64), self.dsp.sample_rate,
                          frame_period=self.dsp.hop_length / self.dsp.sample_rate * 1000)
        if self.dsp.voc_mode == 'RAW':
            quant = self.dsp.encode_mu_law(y, mu=2**self.dsp.bits) \
                if self.dsp.mu_law else self.dsp.float_2_label(y, bits=self.dsp.bits)
        elif self.dsp.voc_mode == 'MOL':
            quant = self.dsp.float_2_label(y, bits=16)
        else:
            raise ValueError(f'Unexpected voc mode {self.dsp.voc_mode}, should be either RAW or MOL.')
        return mel.astype(np.float32), quant.astype(np.int64), pitch.astype(np.float32)


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
args = parser.parse_args()


if __name__ == '__main__':

    config = read_config(args.config)
    wav_files = get_files(args.path, '.wav')
    wav_ids = {w.stem for w in wav_files}
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])
    print(f'\n{len(wav_files)} .wav files found in "{args.path}"')
    assert len(wav_files) > 0, f'Found no wav files in {args.path}, exiting.'

    text_dict = ljspeech(args.path)
    text_dict = {item_id: text for item_id, text in text_dict.items()
                 if item_id in wav_ids and len(text) > config['preprocessing']['min_text_len']}
    wav_files = [w for w in wav_files if w.stem in text_dict]
    print(f'Using {len(wav_files)} wav files that are indexed in metafile.\n')

    n_workers = max(1, args.num_workers)

    dsp = DSP.from_config(config)

    simple_table([
        ('Sample Rate', dsp.sample_rate),
        ('Bit Depth', dsp.bits),
        ('Mu Law', dsp.mu_law),
        ('Hop Length', dsp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}'),
        ('Num Validation', config['preprocessing']['n_val'])
    ])

    pool = Pool(processes=n_workers)
    dataset = []
    cleaned_texts = []
    cleaner = Cleaner.from_config(config)
    preprocessor = Preprocessor(paths=paths,
                                text_dict=text_dict,
                                dsp=dsp,
                                cleaner=cleaner,
                                lang=config['preprocessing']['language'])

    for i, (item_id, length, cleaned_text) in enumerate(pool.imap_unordered(preprocessor, wav_files), 1):
        if item_id in text_dict:
            dataset += [(item_id, length)]
            cleaned_texts += [(item_id, cleaned_text)]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    dataset.sort()
    random = Random(42)
    random.shuffle(dataset)
    train_dataset = dataset[config['preprocessing']['n_val']:]
    val_dataset = dataset[:config['preprocessing']['n_val']]
    # sort val dataset longest to shortest
    val_dataset.sort(key=lambda d: -d[1])
    print(f'First val sample: {val_dataset[0][0]}')

    text_dict = {id: text for id, text in cleaned_texts}

    pickle_binary(text_dict, paths.data/'text_dict.pkl')
    pickle_binary(train_dataset, paths.data/'train_dataset.pkl')
    pickle_binary(val_dataset, paths.data/'val_dataset.pkl')

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
