import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger

from opts import get_opts
from pl_trainer import ACPL_Trainer


def main():
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    args = get_opts()
    cudnn.benchmark = True
    args.distributed = args.world_size > 0 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        if args.multiprocessing_distributed:
            logger.info(f"Use GPU: {args.gpu} for training", enqueue=True)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    trainer = ACPL_Trainer(args)
    trainer.pipeline_master(args)


if __name__ == "__main__":
    main()
