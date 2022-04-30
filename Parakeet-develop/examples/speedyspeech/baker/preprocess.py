# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import itemgetter
from typing import Any
from typing import Dict
from typing import List

import argparse
import jsonlines
import librosa
import logging
import numpy as np
import re
import tqdm
from concurrent.futures import ThreadPoolExecutor
from parakeet.data.get_feats import LogMelFBank
from pathlib import Path
from praatio import tgio

from config import get_cfg_default
from tg_utils import validate_textgrid


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     alignment_fp: Path,
                     output_dir: Path,
                     mel_extractor=None):
    utt_id = fp.stem

    # reading
    y, sr = librosa.load(str(fp), sr=config.sr)  # resampling may occur
    assert len(y.shape) == 1, f"{utt_id} is not a mono-channel audio."
    assert np.abs(
        y).max() <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
    duration = librosa.get_duration(y, sr=sr)

    # intervals with empty lables are ignored
    alignment = tgio.openTextgrid(alignment_fp)

    # validate text grid against audio file
    num_samples = y.shape[0]
    validate_textgrid(alignment, num_samples, sr)

    # only with baker's annotation
    intervals = alignment.tierDict[alignment.tierNameList[0]].entryList

    first, last = intervals[0], intervals[-1]
    if not (first.label == "sil" and first.end < duration):
        logging.warning(
            f" There is something wrong with the fisrt interval {first} in utterance: {utt_id}"
        )
    if not (last.label == "sil" and last.start < duration):
        logging.warning(
            f" There is something wrong with the last interval {last} in utterance: {utt_id}"
        )

    # extract mel feats
    logmel = mel_extractor.get_log_mel_fbank(y)

    # extract phone and duration
    phones = []
    tones = []
    ends = []
    durations_sec = []

    for interval in intervals:
        label = interval.label
        label = label.replace("sp1", "sp")  # Baker has sp1 rather than sp

        # split tone from finals
        match = re.match(r'^(\w+)([012345])$', label)
        if match:
            phones.append(match.group(1))
            tones.append(match.group(2))
        else:
            phones.append(label)
            tones.append('0')
        end = min(duration, interval.end)
        ends.append(end)
        durations_sec.append(end - interval.start)  # duration in seconds

    frame_pos = librosa.time_to_frames(
        ends, sr=sr, hop_length=config.hop_length)
    durations_frame = np.diff(frame_pos, prepend=0)

    num_frames = logmel.shape[0]  # number of frames of the spectrogram
    extra = np.sum(durations_frame) - num_frames
    assert extra <= 0, (
        f"Number of frames inferred from alignemnt is "
        f"larger than number of frames of the spectrogram by {extra} frames")
    durations_frame[-1] += (-extra)

    assert np.sum(durations_frame) == num_frames
    durations_frame = durations_frame.tolist()

    mel_path = output_dir / (utt_id + "_feats.npy")
    np.save(mel_path, logmel)  # (num_frames, n_mels)
    record = {
        "utt_id": utt_id,
        "phones": phones,
        "tones": tones,
        "num_phones": len(phones),
        "num_frames": num_frames,
        "durations": durations_frame,
        "feats": mel_path,  # Path object
    }
    return record


def process_sentences(config,
                      fps: List[Path],
                      alignment_fps: List[Path],
                      output_dir: Path,
                      mel_extractor=None,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp, alignment_fp in tqdm.tqdm(
                zip(fps, alignment_fps), total=len(fps)):
            results.append(
                process_sentence(config, fp, alignment_fp, output_dir,
                                 mel_extractor))
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp, alignment_fp in zip(fps, alignment_fps):
                    future = pool.submit(process_sentence, config, fp,
                                         alignment_fp, output_dir,
                                         mel_extractor)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    results.append(ft.result())

    results.sort(key=itemgetter("utt_id"))
    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.jsonl"
    # NOTE: use relative path to the meta jsonlines file
    with jsonlines.open(metadata_path, 'w') as writer:
        for item in results:
            item["feats"] = str(item["feats"].relative_to(output_dir))
            writer.write(item)
    print("Done")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")
    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to baker dataset.")
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")
    parser.add_argument(
        "--config", type=str, help="yaml format configuration file.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)")
    parser.add_argument(
        "--num_cpu", type=int, default=1, help="number of process.")
    args = parser.parse_args()

    C = get_cfg_default()
    if args.config:
        C.merge_from_file(args.config)
        C.freeze()

    if args.verbose > 1:
        print(vars(args))
        print(C)

    root_dir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    dumpdir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list((root_dir / "Wave").rglob("*.wav")))
    alignment_files = sorted(
        list((root_dir / "PhoneLabeling").rglob("*.interval")))

    # filter out several files that have errors in annotation
    exclude = {'000611', '000662', '002365', '005107'}
    wav_files = [f for f in wav_files if f.stem not in exclude]
    alignment_files = [f for f in alignment_files if f.stem not in exclude]

    # split data into 3 sections
    num_train = 9800
    num_dev = 100

    train_wav_files = wav_files[:num_train]
    dev_wav_files = wav_files[num_train:num_train + num_dev]
    test_wav_files = wav_files[num_train + num_dev:]

    train_alignment_files = alignment_files[:num_train]
    dev_alignment_files = alignment_files[num_train:num_train + num_dev]
    test_alignment_files = alignment_files[num_train + num_dev:]

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    mel_extractor = LogMelFBank(
        sr=C.sr,
        n_fft=C.n_fft,
        hop_length=C.hop_length,
        win_length=C.win_length,
        window=C.window,
        n_mels=C.n_mels,
        fmin=C.fmin,
        fmax=C.fmax)

    # process for the 3 sections
    process_sentences(
        C,
        train_wav_files,
        train_alignment_files,
        train_dump_dir,
        mel_extractor=mel_extractor,
        nprocs=args.num_cpu)
    process_sentences(
        C,
        dev_wav_files,
        dev_alignment_files,
        dev_dump_dir,
        mel_extractor=mel_extractor,
        nprocs=args.num_cpu)
    process_sentences(
        C,
        test_wav_files,
        test_alignment_files,
        test_dump_dir,
        mel_extractor=mel_extractor,
        nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
