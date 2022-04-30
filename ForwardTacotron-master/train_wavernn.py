import argparse

import numpy as np
import torch
from torch import optim

from models.fatchord_version import WaveRNN
from trainer.voc_trainer import VocTrainer
from utils.checkpoints import restore_checkpoint
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Using device:', device)
    print('\nInitialising Model...\n')
    voc_model = WaveRNN.from_config(config).to(device)
    dsp = DSP.from_config(config)
    assert np.cumprod(config['vocoder']['model']['upsample_factors'])[-1] == dsp.hop_length

    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint(model=voc_model, optim=optimizer,
                       path=paths.voc_checkpoints / 'latest_model.pt',
                       device=device)

    voc_trainer = VocTrainer(paths=paths, dsp=dsp, config=config)
    voc_trainer.train(voc_model, optimizer, train_gta=args.gta)