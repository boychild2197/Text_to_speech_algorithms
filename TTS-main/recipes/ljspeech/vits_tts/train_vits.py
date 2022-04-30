import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs, init_training
from TTS.tts.configs import BaseDatasetConfig, VitsConfig

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "../LJSpeech-1.1/")
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)
config = VitsConfig(
    audio=audio_config,
    run_name="vits_ljspeech",
    batch_size=48,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
)
args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
trainer = Trainer(args, config, output_path, c_logger, tb_logger, cudnn_benchmark=True)
trainer.fit()
