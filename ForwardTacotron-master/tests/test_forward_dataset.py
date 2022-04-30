import os
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path

import torch

from utils.dataset import ForwardDataset
from utils.text.tokenizer import Tokenizer


class TestForwardDataset(unittest.TestCase):

    def setUp(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix='TestForwardDatasetTmp')
        self.temp_dir = Path(temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_items(self) -> None:
        text_dict = {'0': 'a', '1': 'bc'}
        data_dir = self.temp_dir / 'data'
        mel_dir = data_dir / 'mel'
        alg_dir = data_dir / 'alg'
        pitch_dir = data_dir / 'phon_pitch'
        energy_dir = data_dir / 'phon_energy'
        mel_dir.mkdir(parents=True)
        alg_dir.mkdir(parents=True)
        pitch_dir.mkdir(parents=True)
        energy_dir.mkdir(parents=True)

        mels = [np.full((2, 2), fill_value=1), np.full((2, 3), fill_value=2)]
        durs = [np.full(1, fill_value=2), np.full(2, fill_value=3)]
        pitches = [np.full(1, fill_value=5), np.full(2, fill_value=6)]
        energies = [np.full(1, fill_value=6), np.full(2, fill_value=7)]

        np.save(mel_dir / '0.npy', mels[0])
        np.save(mel_dir / '1.npy', mels[1])
        np.save(alg_dir / '0.npy', durs[0])
        np.save(alg_dir / '1.npy', durs[1])
        np.save(pitch_dir / '0.npy', pitches[0])
        np.save(pitch_dir / '1.npy', pitches[1])
        np.save(energy_dir / '0.npy', energies[0])
        np.save(energy_dir / '1.npy', energies[1])

        dataset = ForwardDataset(path=data_dir,
                                 dataset_ids=['0', '1'],
                                 text_dict=text_dict,
                                 tokenizer=Tokenizer())

        data = [dataset[i] for i in range(len(dataset))]

        np.testing.assert_allclose(data[0]['mel'], mels[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['mel'], mels[1], rtol=1e-10)
        np.testing.assert_allclose(data[0]['dur'], durs[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['dur'], durs[1], rtol=1e-10)
        np.testing.assert_allclose(data[0]['pitch'], pitches[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['pitch'], pitches[1], rtol=1e-10)
        np.testing.assert_allclose(data[0]['energy'], energies[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['energy'], energies[1], rtol=1e-10)

        self.assertEqual(1, data[0]['x_len'])
        self.assertEqual(2, data[1]['x_len'])
        self.assertEqual('0', data[0]['item_id'])
        self.assertEqual('1', data[1]['item_id'])
        self.assertEqual(2, data[0]['mel_len'])
        self.assertEqual(3, data[1]['mel_len'])
