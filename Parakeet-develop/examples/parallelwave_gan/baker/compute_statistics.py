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
"""Calculate statistics of feature files."""

import argparse
import logging
import os

import jsonlines
import numpy as np
from parakeet.datasets.data_table import DataTable
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import get_cfg_default


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Compute mean and variance of dumped raw features.")
    parser.add_argument(
        "--metadata", type=str, help="json file with id and file paths ")
    parser.add_argument(
        "--field-name",
        type=str,
        help="name of the field to compute statistics for.")
    parser.add_argument(
        "--config", type=str, help="yaml format configuration file.")
    parser.add_argument(
        "--dumpdir",
        type=str,
        help="directory to save statistics. if not provided, "
        "stats will be saved in the above root directory. (default=None)")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )
        logging.warning('Skip DEBUG/INFO messages')

    config = get_cfg_default()
    # load config
    if args.config:
        config.merge_from_file(args.config)

    # check directory existence
    if args.dumpdir is None:
        args.dumpdir = os.path.dirname(args.metadata)
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    with jsonlines.open(args.metadata, 'r') as reader:
        metadata = list(reader)
    dataset = DataTable(
        metadata,
        fields=[args.field_name],
        converters={args.field_name: np.load}, )
    logging.info(f"The number of files = {len(dataset)}.")

    # calculate statistics
    scaler = StandardScaler()
    for datum in tqdm(dataset):
        # StandardScalar supports (*, num_features) by default
        scaler.partial_fit(datum[args.field_name])

    stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
    np.save(
        os.path.join(args.dumpdir, "stats.npy"),
        stats.astype(np.float32),
        allow_pickle=False)


if __name__ == "__main__":
    main()
