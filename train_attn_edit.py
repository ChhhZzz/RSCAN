"""
This file runs the main training/val loop
"""

import json
import pprint
import shutil
import sys
import warnings
import time

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions  # noqa: E402

opts = TrainOptions().parse()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

from training.coach_edit import Coach  # noqa: E402
from options.update_options import update_opts

warnings.filterwarnings("ignore")


def main():
    opts = TrainOptions().parse()
    opts.condition = 'img'
    opts.train_attn = 1
    if opts.exp_dir == 'output':
        opts.exp_dir = 'inversion'
    opts = update_opts(opts)
    # opts.batch_size = 2
    if os.path.exists(opts.exp_dir) and not opts.resume_training:
        shutil.rmtree(opts.exp_dir)
    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, "opt.json"), "w") as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    time.sleep(opts.time)

    coach = Coach(opts)
    coach.train ()
    # coach.train()


if __name__ == "__main__":
    main()
