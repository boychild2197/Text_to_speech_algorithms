#!/bin/bash

stage=0
stop_stage=100
fs=24000
n_shift=300

export MAIN_ROOT=`realpath ${PWD}/../../../`

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # get durations from MFA's result
    echo "Generate durations.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=./aishell3_alignment_tone \
        --output durations.txt \
        --sample-rate=${fs} \
        --n-shift=${n_shift}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # extract features
    echo "Extract features ..."
    python3 ${MAIN_ROOT}/utils/fastspeech2_preprocess.py \
        --dataset=aishell3 \
        --rootdir=~/datasets/data_aishell3/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config-path=conf/default.yaml \
        --num-cpu=8 \
        --cut-sil=True
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="speech"

    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="pitch"

    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="energy"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize and covert phone to id, dev and test should use train's stats
    echo "Normalize ..."
    python3 ${MAIN_ROOT}/utils/fastspeech2_normalize.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --dumpdir=dump/train/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${MAIN_ROOT}/utils/fastspeech2_normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${MAIN_ROOT}/utils/fastspeech2_normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt
fi
