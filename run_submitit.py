# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main_hiera as classification
import submitit
from video_dataset_config import get_dataset_config, DATASET_CONFIG

def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for DeiT", parents=[classification_parser])
    parser.add_argument("--ngpus", default=2, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=5, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="gpu_l40", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    #parser.add_argument('--epochs', default=30, type=int)


    #parser = argparse.ArgumentParser('Hiera training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=18, type=int)
    parser.add_argument('--epochs', default=30, type=int)

    # Dataset parameters
    parser.add_argument('--data_dir', type=str,  metavar='DIR', default='/scratch/datasets/Kinetics400', help='path to dataset')
    parser.add_argument('--dataset', default='kinetics400',
                        choices=list(DATASET_CONFIG.keys()), help='dataset name')
    parser.add_argument('--duration', default=8, type=int, help='number of frames per clip')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='number of frames per group')
    parser.add_argument('--threed_data', action='store_true',
                        help='load data in the layout for 3D conv')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop, directly crop to input_size')
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform deterministic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='data augmentation version')
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
    parser.add_argument('--use_lmdb', action='store_true', help='use lmdb instead of jpeg')
    parser.add_argument('--use_pyav', action='store_true', help='use video directly')

    # Model & Temporal parameters (for transformer models, not used by Hiera)
    parser.add_argument('--pretrained', action='store_true', default=False,
                    help='start with pretrained model')
    # These are ignored for Hiera, but still parsed:
    parser.add_argument('--hpe_to_token', action='store_true', help='(ignored for Hiera)')
    parser.add_argument('--rel_pos', action='store_true', help='(ignored for Hiera)')
    parser.add_argument('--window_size', default=7, type=int, help='(ignored for Hiera)')
    # Update help to indicate this parameter is used for super image creation.
    parser.add_argument('--super_img_rows', default=3, type=int, 
                        help='Number of rows to arrange frames into a super image (if >1, video frames are rearranged into a grid)')

    # Model parameters
    parser.add_argument('--model', default='Hiera', type=str, metavar='MODEL',
                        help='model name (for logging purposes)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='EMA decay')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='Force EMA on CPU')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.5,
                        help='weight decay (default: 0.5)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound (default: 1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='AutoAugment policy (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase probability (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first augmentation split')

    # Mixup parameters
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup/cutmix (default: 1.0)')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both are enabled (default: 0.5)')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='Mixup mode: "batch", "pair", or "elem" (default: "batch")')

    # Output and general training settings
    parser.add_argument('--output_dir', default='/scratch/workspace/sudipta/sifar-pytorch/output',
                        help='Path to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device for training/testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--no-resume-loss-scaler', action='store_false', dest='resume_loss_scaler')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable AMP')
    parser.add_argument('--use_checkpoint', default=False, action='store_true', help='Use checkpoint to save memory')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training setup')
    parser.add_argument('--auto-resume', action='store_true', help='Auto resume')
    parser.add_argument('--simclr_w', type=float, default=0., help='Weight for SimCLR loss')
    parser.add_argument('--contrastive_nomixup', action='store_true', help='Do not use mixup in contrastive learning')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for NCE')
    parser.add_argument('--branch_div_w', type=float, default=0., help='Weight for branch divergence loss')
    parser.add_argument('--simsiam_w', type=float, default=0., help='Weight for SimSiam loss')
    parser.add_argument('--moco_w', type=float, default=0., help='Weight for MoCo loss')
    parser.add_argument('--byol_w', type=float, default=0., help='Weight for BYOL loss')
    parser.add_argument('--finetune', action='store_true', help='Finetune model')
    parser.add_argument('--initial_checkpoint', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--dml_w', type=float, default=0., help='Deep mutual learning weight')
    parser.add_argument('--one_w', type=float, default=0., help='ONE weight')
    parser.add_argument('--kd_temp', type=float, default=1.0, help='Temperature for KD loss')
    parser.add_argument('--mulmix_b', type=float, default=0., help='MulMixture beta')
    parser.add_argument('--hard_contrastive', action='store_true', help='Use hard contrastive loss')
    parser.add_argument('--selfdis_w', type=float, default=0., help='Self distillation weight')
    parser.add_argument('--colorjitter', type=bool, default=False, help='color jitter true ot false')
    parser.add_argument('--local-rank', type=int, help='Local rank for distributed training')
    parser.add_argument('--randomrotation', type=int, default=0, help='rotation value')
    parser.add_argument('--randomarg', type=bool, default=False, help='Random Augment true ot false')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/scratch/workspace/sudipta/Distributed-Heira/sifar-pytorch-heira").is_dir():
        p = Path(f"/scratch/workspace/sudipta/Distributed-Heira/sifar-pytorch-heira/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_hiera as classification

        self._setup_gpu_args()
        
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="deit")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()