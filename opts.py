import argparse
import models.densenet as models


def get_opts():
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    # general setting
    parser = argparse.ArgumentParser(description="ACPL")
    parser.add_argument("--data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="densenet121",
        choices=model_names,
        help="model architecture: "
        + " | ".join(model_names)
        + " (default: densenet121)",
    )

    parser.add_argument(
        "--epochs",
        default=15,
        type=int,
        metavar="N",
        help="number of total epochs to/run",
    )
    parser.add_argument("--eval-interval", default=100, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )

    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.05,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument("--num-class", default=15, type=int)
    # experiment
    parser.add_argument("--task", choices=["cx14", "cxp", "isic"], nargs="*", type=str)

    # semi
    parser.add_argument("--label_ratio", default=2, type=int)
    parser.add_argument("--runtime", default=1, type=int)

    parser.add_argument("--resize", default=256, type=int)

    parser.add_argument("--resume", default=None, type=int)
    parser.add_argument("--total-loops", default=5, type=float)
    parser.add_argument("--desc", default="naive", type=str)
    parser.add_argument("--reinit", action="store_true")
    parser.add_argument("--log-base", default="/mnt/hd/Logs/CPL", type=str)
    parser.add_argument("--pl-epochs", default=5, type=int)
    parser.add_argument("--topk", default=50, type=int)
    parser.add_argument(
        "--sel",
        default=2,
        type=int,
        help="0 for high ds,1 for uncertain, 2 for informative",
    )
    parser.add_argument("--ds-mixup", action="store_true", help="use density mixup")
    parser.add_argument(
        "--num-gmm-sets", default=3, type=int, help="How many sets should gmm divided"
    )
    args = parser.parse_args()
    return args
