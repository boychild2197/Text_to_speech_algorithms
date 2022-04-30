import os
import time
from typing import Tuple, Any, Dict

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.fatchord_version import WaveRNN
from trainer.common import Averager, VocSession, to_device
from utils.checkpoints import save_checkpoint
from utils.dataset import get_vocoder_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table
from utils.distribution import discretized_mix_logistic_loss
from utils.dsp import DSP
from utils.files import unpickle_binary, pickle_binary, get_files, parse_schedule
from utils.paths import Paths


class VocTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.voc_log, comment='v1')
        self.dsp = dsp
        self.config = config
        self.train_cfg = config['vocoder']['training']
        self.loss_func = F.cross_entropy if self.dsp.voc_mode == 'RAW' else discretized_mix_logistic_loss
        path_top_k = paths.voc_top_k/'top_k.pkl'
        if os.path.exists(path_top_k):
            self.top_k_models = unpickle_binary(path_top_k)
            # log recent top models
            for i, (mel_loss, g_wav, m_step, m_name) in enumerate(self.top_k_models, 1):
                self.writer.add_audio(
                    tag=f'Top_K_Models/generated_top_{i}',
                    snd_tensor=g_wav, global_step=m_step, sample_rate=self.dsp.sample_rate)
        else:
            self.top_k_models = []

    def train(self,
              model: WaveRNN,
              optimizer: Optimizer,
              train_gta=False) -> None:
        voc_schedule = self.train_cfg['schedule']
        voc_schedule = parse_schedule(voc_schedule)
        for i, session_params in enumerate(voc_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set, val_set_samples = get_vocoder_datasets(
                    path=self.paths.data, batch_size=bs, train_gta=train_gta,
                    max_mel_len=self.train_cfg['max_mel_len'], hop_length=self.dsp.hop_length,
                    voc_pad=model.pad, voc_seq_len=self.train_cfg['seq_len'],
                    voc_mode=self.dsp.voc_mode, bits=self.dsp.bits,
                    num_gen_samples=self.train_cfg['num_gen_samples'])
                session = VocSession(
                    index=i, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set,
                    val_set_samples=val_set_samples)
                self.train_session(model, optimizer, session, train_gta)

    def train_session(self, model: WaveRNN,
                      optimizer: Optimizer,
                      session: VocSession,
                      train_gta: bool) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps ', str(training_steps // 1000) + 'k'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr),
                      ('Sequence Length', self.train_cfg['seq_len']),
                      ('GTA Training', train_gta)])
        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters

        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                start = time.time()
                model.train()
                batch = to_device(batch, device=device)
                x, y = batch['x'], batch['y']
                y_hat = model(x, batch['mel'])
                if model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif model.mode == 'MOL':
                    y = batch['y'].float()
                y = y.unsqueeze(-1)

                loss = self.loss_func(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optimizer.step()
                loss_avg.add(loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['gen_samples_every'] == 0:
                    stream(msg + 'generating samples...')
                    gen_result = self.generate_samples(model, session)
                    if gen_result is not None:
                        mel_loss, gen_wav = gen_result
                        self.writer.add_scalar('Loss/generated_mel_l1', mel_loss, model.get_step())
                        self.track_top_models(mel_loss, gen_wav, model)

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.voc_checkpoints / f'wavernn_step{k}k.pt')

                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Loss/val', val_loss, model.get_step())
            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.voc_checkpoints / 'latest_model.pt')

            loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: WaveRNN, val_set: Dataset) -> float:
        model.eval()
        val_loss = 0
        device = next(model.parameters()).device
        for i, batch in enumerate(val_set, 1):
            batch = to_device(batch, device=device)
            x, y, m = batch['x'], batch['y'], batch['mel']
            with torch.no_grad():
                y_hat = model(x, m)
                if model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif model.mode == 'MOL':
                    y = y.float()
                y = y.unsqueeze(-1)
                loss = self.loss_func(y_hat, y)
                val_loss += loss.item()
        return val_loss / len(val_set)

    @ignore_exception
    def generate_samples(self,
                         model: WaveRNN,
                         session: VocSession) -> Tuple[float, list]:
        """
        Generates audio samples to cherry-pick models. To evaluate audio quality
        we calculate the l1 distance between mels of predictions and targets.
        """
        model.eval()
        mel_losses = []
        gen_wavs = []
        device = next(model.parameters()).device
        for i, sample in enumerate(session.val_set_samples, 1):
            m, x = sample['mel'], sample['x']
            if i > self.train_cfg['num_gen_samples']:
                break
            x = x[0].numpy()
            bits = 16 if self.dsp.voc_mode == 'MOL' else self.dsp.bits
            if self.dsp.mu_law and self.dsp.voc_mode != 'MOL':
                x = DSP.decode_mu_law(x, 2 ** bits, from_labels=True)
            else:
                x = DSP.label_2_float(x, bits)
            gen_wav = model.generate(
                mels=m, batched=self.train_cfg['gen_batched'],
                target=self.train_cfg['target'], overlap=self.train_cfg['overlap'],
                mu_law=self.dsp.mu_law, silent=True)

            gen_wavs.append(gen_wav)
            y_mel = self.dsp.wav_to_mel(x.squeeze(), normalize=False)
            y_mel = torch.tensor(y_mel).to(device)
            y_hat_mel = self.dsp.wav_to_mel(gen_wav, normalize=False)
            y_hat_mel = torch.tensor(y_hat_mel).to(device)
            loss = F.l1_loss(y_hat_mel, y_mel)
            mel_losses.append(loss.item())

            self.writer.add_audio(
                tag=f'Validation_Samples/target_{i}', snd_tensor=x,
                global_step=model.step, sample_rate=self.dsp.sample_rate)
            self.writer.add_audio(
                tag=f'Validation_Samples/generated_{i}',
                snd_tensor=gen_wav, global_step=model.step, sample_rate=self.dsp.sample_rate)

        return sum(mel_losses) / len(mel_losses), gen_wavs[0]

    def track_top_models(self, mel_loss, gen_wav, model):
        """ Keeps track of top k models and saves them according to their current rank """
        for j, (l, g, m, m_n) in enumerate(self.top_k_models):
            print(f'{j} {l} {m} {m_n}')
        if len(self.top_k_models) < self.train_cfg['keep_top_k'] or mel_loss < self.top_k_models[-1][0]:
            m_step = model.get_step()
            model_name = f'model_loss{mel_loss:#0.5}_step{m_step}_weights.pyt'
            self.top_k_models.append((mel_loss, gen_wav, model.get_step(), model_name))
            self.top_k_models.sort(key=lambda t: t[0])
            self.top_k_models = self.top_k_models[:self.train_cfg['keep_top_k']]
            model.save(self.paths.voc_top_k/model_name)
            all_models = get_files(self.paths.voc_top_k, extension='pyt')
            top_k_names = {m[-1] for m in self.top_k_models}
            for model_file in all_models:
                if model_file.name not in top_k_names:
                    print(f'removing {model_file}')
                    os.remove(model_file)
            pickle_binary(self.top_k_models, self.paths.voc_top_k/'top_k.pkl')

            for i, (mel_loss, g_wav, m_step, m_name) in enumerate(self.top_k_models, 1):
                self.writer.add_audio(
                    tag=f'Top_K_Models/generated_top_{i}',
                    snd_tensor=g_wav, global_step=m_step, sample_rate=self.dsp.sample_rate)
