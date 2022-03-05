import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from easydict import EasyDict as edict
from loguru import logger
from sklearn.metrics import roc_auc_score
from torchvision.transforms import transforms
from tqdm import tqdm

import models.densenet as densenet
import models.resnet as resnet
from data.dataloaders.dataloader_semi import ChestDataloader
from data.dataloaders.dataloaders_isic import ISICDataloader
from pl_utils import PLUL
from utils import distributed_concat


class ACPL_Trainer:
    def __init__(self, args):
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.best_metric = 0
        self.task = args.task[0]
        self.best_iter = 0

        self.best_loop = 0
        self._transform_init(args)
        self._loader_init(args)

        self._log_init(args)

        self._model_init(args)
        self._optim_init(args)

        self._crit_init(args, self.label_loader1)

        self.last_loop_anchor_len = len(self.anchor_dataset)
        if dist.get_rank() == 0:
            print(args)

    def pipeline_master(self, args):
        if args.resume is None:
            total_iter = args.epochs * (
                len(self.label_loader1.dataset) // self.label_loader1.batch_size
            )

            self.model_training(args, self.label_loader1, total_iter, 0)
        else:
            self.best_iter = args.resume
            self.best_loop = 0
        self._ck_load(args, self.best_iter, self.best_loop)
        # self.test(args, self.best_iter)

        anchor = self._anchor_ext(args)
        if dist.get_rank() == 0:
            torch.save(anchor, f"{self.embed_log}/anchor0.pth.tar")
        for i in range(1, args.total_loops + 1):
            if dist.get_rank() == 0:
                logger.info(f"Loop {i} start", enqueue=True)

            unlabel = self._anchor_sim(args)
            if dist.get_rank() == 0:
                torch.save(unlabel, f"{self.embed_log}/ugraph{i - 1}.pth.tar")

            # Build local-KNN graph
            dist.barrier()
            if dist.get_rank() == 1:
                time.sleep(10)
            logger.info(f"Rank {dist.get_rank()} start graph building")
            lpul = PLUL(
                anchor,
                unlabel,
                args,
                self.embed_log,
                ds_mixup=args.ds_mixup,
                loop=i,
                num_gmm_sets=args.num_gmm_sets,
            )

            # label
            l_sel_idxs, l_sel_p = lpul.get_new_label()

            # unlabel
            u_sel_idxs = lpul.get_new_unlabel()

            # anchor
            anchor_idxs = lpul.anchor_purify()
            a_sel_idxs, a_sel_p = lpul.get_new_anchor(anchor_idxs)

            self.label_dataset.x_add_pl(l_sel_p, l_sel_idxs)
            self.label_dataset.u_update_pl(l_sel_idxs)

            self.anchor_dataset.x_add_pl(a_sel_p, a_sel_idxs)
            # self.anchor_dataset.u_update_pl(a_sel_idxs)

            self.unlabel_dataset.u_update_pl(u_sel_idxs)
            logger.info(
                f"Rank {dist.get_rank()} Loop {i} label size {len(self.label_dataset)}, unlabel size {len(self.unlabel_dataset)}"
            )
            del lpul, anchor, unlabel
            dist.barrier()
            (self.label_loader1, self.label_dataset, self.label_sampler1,) = self.loader.run(
                "labeled",
                dataset=self.label_dataset,
                transform=self.train_transform,
                ratio=args.label_ratio,
                runtime=args.runtime,
            )

            (self.unlabel_loader, self.unlabel_dataset, self.unlabel_sampler,) = self.loader.run(
                "unlabeled",
                dataset=self.unlabel_dataset,
                transform=self.test_transform,
                ratio=args.label_ratio,
                runtime=args.runtime,
            )

            self.anchor_loader, self.anchor_dataset, self.sampler = self.loader.run(
                "anchor",
                dataset=self.anchor_dataset,
                transform=self.test_transform,
                ratio=args.label_ratio,
                runtime=args.runtime,
            )
            # train LP
            if args.reinit:
                self._model_init(args)
                self._optim_init(args)
                self._crit_init(args, self.label_loader1)

            total_iter = args.pl_epochs * (
                len(self.label_loader1.dataset) // self.label_loader1.batch_size
            )
            self.model_training(args, self.label_loader1, total_iter, i)

            dist.barrier()
            self._ck_load(args, self.best_iter, self.best_loop)

            if dist.get_rank() == 0:
                logger.info(
                    f"Current pseudo Anchor size {len(self.anchor_dataset)}, Last loop anchor size {self.last_loop_anchor_len}"
                )
            # verify pseudo anchors
            pa_pack = self._anchor_ext(args)  # last loop anchors + info
            self.last_loop_anchor_len = len(self.anchor_dataset)
            anchor = pa_pack
            if dist.get_rank() == 0:
                torch.save(anchor, f"{self.embed_log}/anchor{i}.pth.tar")

    def model_training(self, args, loader, total_iter, loop):
        self.net1.train()
        self.net1_ema.train()

        train_loader_iter1 = iter(loader)
        losses1 = AverageMeter()

        if dist.get_rank() == 0:
            logger.info(f"| Start warmup", enqueue=True)

        for i in range(total_iter):
            if args.distributed:
                loader.sampler.set_epoch(i)
            self.adjust_learning_rate(self.optimizer1, i, total_iter, args)

            try:
                inputs1_x_w, labels1, item1, input_paths = train_loader_iter1.next()
            except:
                train_loader_iter1 = iter(loader)
                inputs1_x_w, labels1, item1, input_paths = train_loader_iter1.next()

            net1_losses = self._train(
                args,
                i,
                self.net1,
                self.optimizer1,
                self.criterion1,
                inputs1_x_w,
                labels1,
                item1,
                net2_ema=self.net1_ema,
                optimizer_ema=self.optimizer1_ema,
            )

            losses1.update(net1_losses["ce"])

            if (i + 1) % args.eval_interval == 0:
                if dist.get_rank() == 0:
                    logger.debug(
                        f"| Train Loop {loop} Iter {i + 1}/{total_iter} || Net1 BCE {losses1.avg}",
                        enqueue=True,
                    )
            if (i + 1) % (args.eval_interval * 10) == 0 or (i + 1) == total_iter:
                all_auc, mean_auc = self.test(args, i + 1)
                state_dict = {
                    "Iter": i + 1,
                    "net1": self.net1_ema.state_dict(),
                    "all_auc": all_auc,
                    "mean_auc": mean_auc,
                }
                ck_name = f"{self.ck_log}/ck_Loop{loop}_Iter{i + 1}.pth.tar"
                if dist.get_rank() == 0:
                    torch.save(state_dict, ck_name)
                if mean_auc > self.best_metric:
                    self.best_metric = mean_auc
                    self.best_iter = i + 1
                    self.best_loop = loop
                    state_dict["best"] = True
                    ck_name = f"{self.best_ck_path}_Loop{loop}_Iter{i + 1}.pth.tar"
                    if dist.get_rank() == 0:
                        torch.save(state_dict, ck_name)

    def _anchor_sim(self, args):
        self.net1_ema.eval()

        u_gts, u_idxs = torch.tensor([]).cuda(args.gpu), torch.tensor([]).cuda(args.gpu)
        u_preds1 = torch.tensor([]).cuda(args.gpu)
        u_embed1 = torch.tensor([]).cuda(args.gpu)
        u_logits1 = torch.tensor([]).cuda(args.gpu)
        u_paths = torch.tensor([]).cuda(args.gpu)

        with torch.no_grad():
            for batch_idx, (inputs, labels, item, input_path) in enumerate(
                tqdm(self.unlabel_loader)
            ):
                inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
                item = item.cuda(args.gpu)
                input_path = input_path.cuda(args.gpu)

                outputs1, feat1 = self.net1_ema(inputs)
                u_embed1 = torch.cat((u_embed1, feat1))

                u_idxs = torch.cat((u_idxs, item))
                u_paths = torch.cat((u_paths, input_path))
                u_gts = torch.cat((u_gts, labels))

                u_preds1 = torch.cat((u_preds1, torch.sigmoid(outputs1)))
                u_logits1 = torch.cat((u_logits1, outputs1))

        length = len(self.unlabel_loader.dataset)
        u_idxs = distributed_concat(u_idxs.contiguous(), length)
        u_gts = distributed_concat(u_gts.contiguous(), length)
        u_preds1 = distributed_concat(u_preds1.contiguous(), length)
        u_logits1 = distributed_concat(u_logits1.contiguous(), length)
        u_embed1 = distributed_concat(u_embed1.contiguous(), length)
        u_paths = distributed_concat(u_paths.contiguous(), length)

        return edict(
            {
                "idxs": u_idxs,
                "gts": u_gts,
                "p1": u_preds1,
                "embed1": u_embed1,
                "logits1": u_logits1,
                "path": u_paths,
            }
        )

    def _anchor_ext(self, args):
        self.net1_ema.eval()
        p1, logits1, embed1, gts, idxs = (
            torch.tensor([]).cuda(args.gpu),
            torch.tensor([]).cuda(args.gpu),
            torch.tensor([]).cuda(args.gpu),
            torch.tensor([]).cuda(args.gpu),
            torch.tensor([]).cuda(args.gpu),
        )
        with torch.no_grad():
            for batch_idx, (inputs, labels, item, _) in enumerate(tqdm(self.anchor_loader)):
                inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
                # item = torch.from_numpy(item.numpy()).cuda(args.gpu)

                outputs1, feat1 = self.net1_ema(inputs)
                item = torch.from_numpy(item.numpy()).cuda(args.gpu)
                embed1 = torch.cat((embed1, feat1))
                idxs = torch.cat((idxs, item))
                gts = torch.cat((gts, labels))
                logits1 = torch.cat((logits1, outputs1))
                p1 = torch.cat((p1, torch.sigmoid(outputs1)))
        length = len(self.anchor_loader.dataset)
        p1 = distributed_concat(p1.contiguous(), length)
        logits1 = distributed_concat(logits1.contiguous(), length)
        embed1 = distributed_concat(embed1.contiguous(), length)
        gts = distributed_concat(gts.contiguous(), length)
        idxs = distributed_concat(idxs.contiguous(), length)
        return edict(
            {
                "embed1": embed1,
                "idxs": idxs,
                "gts": gts,
                "p1": p1,
                "logits1": logits1,
            }
        )

    def _train(
        self,
        args,
        i,
        net,
        optimizer,
        criterion,
        inputs,
        labels,
        item,
        net2_ema=None,
        optimizer_ema=None,
    ):
        inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
        optimizer.zero_grad()

        outputs = self._iter_forward(inputs, net, net2_ema)
        losses = self._iter_loss(outputs["s"], labels, criterion)
        total_loss = losses["ce"].mean()

        self.scaler.scale(total_loss).backward()
        self.scaler.step(optimizer)
        optimizer_ema.step()
        self.scaler.update()

        loss_val = {"ce": losses["ce"].mean().item()}
        return loss_val

    def _iter_forward(self, inputs, net, net2_ema=None):
        with torch.cuda.amp.autocast(enabled=True):
            outputs_s, _ = net(inputs)
            if net2_ema:
                outputs_t, _ = net2_ema(inputs)
                return {"s": outputs_s, "t": outputs_t}
            else:
                return {"s": outputs_s}

    def _iter_loss(self, outputs, labels, criterion):
        loss = criterion(outputs.float(), labels.float())
        return {"ce": loss}

    def _ck_load(self, args, i, loop):
        logger.info(f"Rank {dist.get_rank()} Loading from Loop {loop} Iter {i}")
        load_path = f"{self.best_ck_path}_Loop{loop}_Iter{i}.pth.tar"

        ck = torch.load(load_path, map_location="cpu")
        state_dict1 = ck["net1"]
        self.net1_ema.load_state_dict(state_dict1, strict=False)
        del ck, state_dict1
        with torch.no_grad():
            torch.cuda.empty_cache()

    def test(self, args, i):
        self.net1_ema.eval()
        targs, preds = torch.LongTensor([]).cuda(args.gpu), torch.tensor([]).cuda(args.gpu)
        if dist.get_rank() == 0:
            logger.info(f"Start Testing Iteration {i}", enqueue=True)
        with torch.no_grad():
            for batch_idx, (inputs, targets, item, _) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
                with torch.cuda.amp.autocast():
                    outputs1, _ = self.net1_ema(inputs)
                outputs = outputs1

                preds = torch.cat((preds, torch.sigmoid(outputs)))
                targs = torch.cat((targs, targets))

        preds = distributed_concat(preds, len(self.test_loader.dataset)).cpu().numpy()
        targs = distributed_concat(targs, len(self.test_loader.dataset)).cpu().numpy()
        all_auc = np.asarray(
            [roc_auc_score(targs[:, i], preds[:, i]) for i in range(args.num_class - 1)],
        )
        mean_auc = all_auc.mean()
        if dist.get_rank() == 0:
            logger.critical(f"Iteration {i} Mean AUC {mean_auc}", enqueue=True)
        return all_auc, mean_auc

    def _log_init(self, args):
        log_base = f"{args.log_base}/{args.task[0]}/{args.desc}"
        Path(log_base).mkdir(parents=True, exist_ok=True)
        self.info_log = os.path.join(log_base, "info.log")
        open(self.info_log, "w")
        logger.add(self.info_log, enqueue=True)

        self.ck_log = os.path.join(log_base, "checkpoints")
        Path(self.ck_log).mkdir(parents=True, exist_ok=True)
        self.best_ck_path = os.path.join(log_base, "model_best")

        self.embed_log = os.path.join(log_base, "embed")
        Path(self.embed_log).mkdir(parents=True, exist_ok=True)

    def _transform_init(self, args):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.resize, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(args.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _loader_init(self, args):
        if self.task == "cx14":
            self.loader = ChestDataloader(
                args.batch_size,
                args.num_workers,
                args.resize,
                args.data,
            )
        elif self.task == "isic":
            self.loader = ISICDataloader(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_resize=args.img_resize,
                root_dir=args.data,
            )
        self.test_loader, self.test_dataset, self.test_sampler = self.loader.run(
            "test",
            transform=self.test_transform,
            ratio=args.label_ratio,
            runtime=args.runtime,
        )
        (self.label_loader1, self.label_dataset, self.label_sampler1,) = self.loader.run(
            "labeled",
            transform=self.train_transform,
            ratio=args.label_ratio,
            runtime=args.runtime,
        )

        (self.anchor_loader, self.anchor_dataset, self.anchor_sampler,) = self.loader.run(
            "anchor",
            transform=self.test_transform,
            ratio=args.label_ratio,
            runtime=args.runtime,
        )

        if args.label_ratio != 100:
            (self.unlabel_loader, self.unlabel_dataset, self.unlabel_sampler,) = self.loader.run(
                "unlabeled",
                transform=self.test_transform,
                ratio=args.label_ratio,
                runtime=args.runtime,
            )
        else:
            unlabel_loader = None

    def _crit_init(self, args, loader, num_examp=None):
        self.criterion1 = nn.BCEWithLogitsLoss(reduction="none").cuda(args.gpu)

    def _create_model(self, arch, num_class, ema=False, imagenet=True):
        model1 = arch(pretrained=imagenet)
        in_features = 1024
        model1.classifier = nn.Linear(in_features, num_class)
        # model1.fc = nn.Linear(in_features, num_class)

        model1 = nn.SyncBatchNorm.convert_sync_batchnorm(model1)
        return model1

    def _model_init(self, args):
        backbone = densenet.densenet121
        # backbone = densenet.densenet169
        # backbone = resnet.resnet50
        self.net1 = self._create_model(backbone, args.num_class)
        self.net1_ema = self._create_model(
            backbone,
            args.num_class,
        )

        if args.distributed:
            if args.gpu is not None:
                self.net1.cuda(args.gpu)
                self.net1_ema.cuda(args.gpu)
                self.net1 = torch.nn.parallel.DistributedDataParallel(
                    self.net1, device_ids=[args.gpu]
                )
                self.net1_ema = torch.nn.parallel.DistributedDataParallel(
                    self.net1_ema, device_ids=[args.gpu]
                )

    def _optim_init(self, args):

        self.optimizer1 = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.net1.parameters())),
            lr=args.lr,
            betas=(0.9, 0.99),
            eps=0.1,
        )
        self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)

        for param in self.net1_ema.parameters():
            param.detach_()

    def adjust_learning_rate(self, optimizer, iter, total_iter, args):
        """Decay the learning rate based on schedule"""
        if (iter / total_iter) <= 0.7:
            lr = args.lr
        elif (iter / total_iter) > 0.7:
            lr = args.lr * 0.1
        # lr *= 0.5 * (1.0 + math.cos(math.pi * iter / total_iter))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.99):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = model.module.state_dict()
        self.ema_params = ema_model.module.state_dict()
        # self.wd = 0.02 * args.lr

        for (k, param), (ema_k, ema_param) in zip(self.params.items(), self.ema_params.items()):
            ema_param.data.copy_(param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for (k, param), (ema_k, ema_param) in zip(self.params.items(), self.ema_params.items()):
            if param.type() == "torch.cuda.LongTensor":
                ema_param = param
            else:
                # if "num_batches_tracked" in k:
                #     ema_param.copy_(param)
                # else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = float("{:.6f}".format(self.sum / self.count))
