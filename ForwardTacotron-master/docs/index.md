Inspired by Microsoft's [FastSpeech](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)
we modified Tacotron to generate speech in a single forward pass using a duration predictor to align text and generated mel spectrograms.

## NEW (14.05.2021): Forward Tacotron V2 (Energy + Pitch) + HiFiGAN Vocoder

The samples are generated with a model trained 80K steps on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) together with the pretrained HiFiGAN vocoder provided by the [HiFiGAN repo](https://github.com/jik876/hifi-gan).

<p class="text">Scientists at the CERN laboratory say they have discovered a new particle.</p> 

<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_hifigan_80k.wav?raw=true" controls preload></audio>

<p class="text">There’s a way to measure the acute emotional intelligence that has never gone out of style.</p>

<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_hifigan_80k.wav?raw=true" controls preload></audio>

<p class="text">President Trump met with other leaders at the Group of 20 conference.</p>

<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_hifigan_80k.wav?raw=true" controls preload></audio>

<p class="text">In a statement announcing his resignation, Mr Ross, said: "While the intentions may have been well meaning, the reaction to this news shows that Mr Cummings interpretation of the government advice was not shared by the vast majority of people who have done as the government asked."</p>

<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/8_hifigan_80k.wav?raw=true" controls preload></audio>


## Forward Tacotron + MelGAN Vocoder

The samples are generated with a model trained 400K steps on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) together with the pretrained MelGAN vocoder provided by the [MelGAN repo](https://github.com/seungwonpark/melgan).

<p class="text">Scientists at the CERN laboratory say they have discovered a new particle.</p> 

| normal speed | faster (1.25) | slower (0.85) |
|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_melgan_400k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_melgan_400k_1.25.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_melgan_400k_0.85.wav?raw=true" controls preload></audio>|

<p class="text">There’s a way to measure the acute emotional intelligence that has never gone out of style.</p>

|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_melgan_400k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_melgan_400k_1.25.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_melgan_400k_0.85.wav?raw=true" controls preload></audio>|

<p class="text">President Trump met with other leaders at the Group of 20 conference.</p>

|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_melgan_400k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_melgan_400k_1.25.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_melgan_400k_0.85.wav?raw=true" controls preload></audio>|

## Forward Tacotron + WaveRNN Vocoder

The samples are generated with a model trained 100K steps on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) together with the pretrained WaveRNN vocoder provided by the [WaveRNN repo](https://github.com/fatchord/WaveRNN).

<p class="text">Scientists at the CERN laboratory say they have discovered a new particle.</p> 

| normal speed | faster (1.25) | slower (0.8) |
|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_wavernn_batched_100k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_wavernn_batched_100k_1.25.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/1_wavernn_batched_100k_0.8.wav?raw=true" controls preload></audio>|

<p class="text">There’s a way to measure the acute emotional intelligence that has never gone out of style.</p>

|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_wavernn_batched_100k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_wavernn_batched_100k_1.25.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/2_wavernn_batched_100k_0.8.wav?raw=true" controls preload></audio>|


<p class="text">President Trump met with other leaders at the Group of 20 conference.</p>

|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_wavernn_batched_100k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_wavernn_batched_100k_1.25.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/3_wavernn_batched_100k_0.8.wav?raw=true" controls preload></audio>|

## Forward Tacotron + Griffin-Lim

<p class="text">The Senate's bill to repeal and replace the Affordable Care-Act is now imperiled.</p>

| normal speed | faster (1.4) | slower (0.6) |
|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/4_griffinlim_100k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/4_griffinlim_100k_1.4.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/4_griffinlim_100k_0.6.wav?raw=true" controls preload></audio>|

<p class="text">Generative adversarial network or variational auto-encoder.</p>

|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/5_griffinlim_100k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/5_griffinlim_100k_1.4.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/5_griffinlim_100k_0.6.wav?raw=true" controls preload></audio>|

<p class="text">Basilar membrane and otolaryngology are not auto-correlations.</p>

|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/6_griffinlim_100k.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/6_griffinlim_100k_1.4.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/6_griffinlim_100k_0.6.wav?raw=true" controls preload></audio>|

&nbsp;
<p class="text">Synthetic speech can be created by concatenating pieces of recorded speech that are stored in a database. Systems differ in the size of the stored speech units; a system that stores phones or diphones provides the largest output range, but may lack clarity. For specific usage domains, the storage of entire words or sentences allows for high-quality output. Alternatively, a synthesizer can incorporate a model of the vocal tract and other human voice characteristics to create a completely "synthetic" voice output.</p>
<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/long_griffinlim.wav?raw=true" controls preload></audio>