# TransformerTTS with LJSpeech

## Dataset

### Download the datasaet.

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
```

### Extract the dataset.

```bash
tar xjvf LJSpeech-1.1.tar.bz2
```

### Preprocess the dataset.

Assume the path to save the preprocessed dataset is `ljspeech_transformer_tts`. Run the command below to preprocess the dataset.

```bash
python preprocess.py --input=LJSpeech-1.1/  --output=ljspeech_transformer_tts
```

## Train the model

The training script requires 4 command line arguments.
`--data` is the path of the training dataset, `--output` is the path of the output direcctory (we recommend to use a subdirectory in `runs` to manage different experiments.)

`--device` should be "cpu" or "gpu", `--nprocs` is the number of processes to train the model in parallel.

```bash
python train.py --data=ljspeech_transformer_tts/ --output=runs/test --device="gpu" --nprocs=1
```

If you want distributed training, set a larger `--nprocs` (e.g. 4). Note that distributed training with cpu is not supported yet.

## Synthesize

Synthesize waveform. We assume the `--input` is a text file, one sentence per line, and `--output` is a directory to save the synthesized mel spectrogram(log magnitude) in `.npy` format. The mel spectrograms can be used with `Waveflow` to generate waveforms.

`--checkpoint_path` should be the path of the parameter file (`.pdparams`) to load. Note that the extention name `.pdparmas` is not included here.

`--device` specifies to device to run synthesis on.

```bash
python synthesize.py --input=sentence.txt --output=mels/ --checkpoint_path='step-310000' --device="gpu" --verbose
```

## Pretrained Model

Pretrained model can be downloaded here. [transformer_tts_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.3.zip).
