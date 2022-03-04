import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from loguru import logger
from sklearn.mixture import GaussianMixture


class PLUL:
    def __init__(
        self, x_info, u_info, args, log_path, ds_mixup=True, loop=0, num_gmm_sets=3
    ) -> None:
        self.x_info = x_info
        self.u_info = u_info
        self.log_path = log_path
        self.ds_mixup = ds_mixup
        self.loop = loop

        self.local_info = self.build_local_graph(
            x_info["embed1"].cpu().numpy(),
            u_info["embed1"].cpu().numpy(),
            args,
        )

        self.local_ds = self.get_ds()
        self.idxs_pack = self.build_GMM(num_gmm_sets=num_gmm_sets, fig=True)

        self.sel = self.idxs_pack[args.sel]
        self.local_agg = self.get_agg()

        self.local_info_r = self.build_local_graph(
            u_info["embed1"].cpu().numpy(), x_info["embed1"].cpu().numpy(), args
        )

    def anchor_purify(self):
        result_idxs = []
        for i in range(self.local_info["i"][self.sel].shape[0]):
            unlabel_node = self.local_info["i"][self.sel[i]]
            mask = np.asarray(
                [
                    self.sel[i] in self.local_info_r["i"][unlabel_node[j]]
                    for j in range(unlabel_node.shape[0])
                ]
            )

            mask_idxs = mask.nonzero()[0]
            result_idxs.append(mask_idxs)

        result_mask = np.asarray([i.shape[0] for i in result_idxs])
        anchor_idxs = self.sel[(result_mask == result_mask.min()).nonzero()[0]]
        return anchor_idxs

    def build_local_graph(self, x_embed, u_embed, args):
        # GPU
        index_ulf = faiss.IndexFlatIP(x_embed.shape[1])
        res_ulf = faiss.StandardGpuResources()
        # gpu_index_ulf = faiss.index_cpu_to_all_gpus(index_ulf)
        gpu_index_ulf = faiss.index_cpu_to_gpu(res_ulf, args.gpu, index_ulf)
        gpu_index_ulf.add(x_embed)
        # index_ulf.add(x_embed)
        D_ulf, I_ulf = gpu_index_ulf.search(u_embed, args.topk)
        # D_ulf, I_ulf = index_ulf.search(u_embed, args.topk)
        # del gpu_index_ulf
        return edict({"d": D_ulf, "i": I_ulf})

        # CPU
        index_ulf = faiss.IndexFlatIP(x_embed.shape[1])
        # res_ulf = faiss.StandardGpuResources()
        # gpu_index_ulf = faiss.index_cpu_to_all_gpus(index_ulf)
        # gpu_index_ulf = faiss.index_cpu_to_gpu(res_ulf, args.gpu, index_ulf)
        # gpu_index_ulf.add(x_embed)
        index_ulf.add(x_embed)
        # D_ulf, I_ulf = gpu_index_ulf.search(u_embed, args.topk)
        D_ulf, I_ulf = index_ulf.search(u_embed, args.topk)
        # del gpu_index_ulf
        return edict({"d": D_ulf, "i": I_ulf})


    def get_ds(self):
        ds = self.local_info["d"].mean(1)
        ds = (ds - ds.min()) / (ds.max() - ds.min())
        ds = ds.reshape(-1, 1)
        return ds

    def get_agg(self):
        knn_gts = self.x_info["gts"][self.local_info["i"]].cpu().numpy()
        agg = knn_gts.mean(1)
        return edict({"knn_gts": knn_gts, "agg": agg})

    def get_pseudo(self, target, val=False, num_class=14):
        pred = self.u_info["p1"][target].cpu().numpy()
        agg = self.local_agg["agg"][target]
        if self.ds_mixup:
            weight = self.local_ds[target]
        else:
            # weight = np.random.beta(1.0,1.0)
            weight = 1.0
        mix = weight * pred + (1 - weight) * agg
        gts = self.u_info["gts"][target].cpu().numpy()
        if val:
            mix_auc = [
                auc_roc_score(mix[:, i].squeeze(), gts.squeeze()[:, i]) for i in range(num_class)
            ]

            pl_auc = [
                auc_roc_score(pred[:, i].squeeze(), gts.squeeze()[:, i]) for i in range(num_class)
            ]

            agg_auc = [
                auc_roc_score(agg[:, i].squeeze(), gts.squeeze()[:, i]) for i in range(num_class)
            ]
            mix_auc_mean = torch.stack(mix_auc).mean()
            pl_auc_mean = torch.stack(pl_auc).mean()
            agg_auc_mean = torch.stack(agg_auc).mean()

            if dist.get_rank() == 0:
                logger.info(
                    f"Mixing {mix_auc_mean} \n PL {pl_auc}, {pl_auc_mean} \n AGG {agg_auc}, {agg_auc_mean}"
                )

        return mix

    def get_new_label(self):
        sel_idxs = self.u_info["idxs"][self.sel].cpu().numpy().astype(int)
        sel_pseudo = self.get_pseudo(self.sel)
        return sel_idxs, sel_pseudo

    def get_new_anchor(self, sel):
        sel_idxs = self.u_info["idxs"][sel].cpu().numpy().astype(int)
        sel_pseudo = self.get_pseudo(sel)
        return sel_idxs, sel_pseudo

    def get_new_unlabel(self):
        return self.u_info["idxs"][self.sel].cpu().numpy().astype(int)

    def build_GMM(self, num_gmm_sets=3, fig=False, name="Local Density"):
        target = self.local_ds
        gmm1 = GaussianMixture(
            n_components=num_gmm_sets,
            max_iter=20,
            tol=1e-2,
            reg_covar=5e-7,
            random_state=1,
        )

        gmm1.fit(target)
        pred = gmm1.predict(target)
        info_idx = pred[target.argmin()]
        high_idx = pred[target.argmax()]
        high_target = (pred == high_idx).nonzero()[0]
        info_target = (pred == info_idx).nonzero()[0]

        if num_gmm_sets == 3:
            uncertain_idx = 3 - info_idx - high_idx
            uncertain_target = (pred == uncertain_idx).nonzero()[0]

        if fig and dist.get_rank() == 0:
            plt.hist(
                target[high_target],
                bins=200,
                range=(0.0, 1.0),
                edgecolor="black",
                alpha=0.5,
                label=f"High {name}",
            )
            plt.hist(
                target[info_target],
                bins=200,
                range=(0.0, 1.0),
                edgecolor="black",
                alpha=0.5,
                label=f"Informative {name}",
            )
            if num_gmm_sets == 3:
                plt.hist(
                    target[uncertain_target],
                    bins=200,
                    range=(0.0, 1.0),
                    edgecolor="black",
                    alpha=0.5,
                    label=f"Uncertain {name}",
                )
            plt.legend()
            plt.grid()
            plt.savefig(f"{self.log_path}/{name}_{self.loop}")
            plt.clf()
        if num_gmm_sets == 3:
            return high_target, uncertain_target, info_target
        else:
            return high_target, info_target


def auc_roc_score(input, targ):
    "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
    fpr, tpr = roc_curve(input, targ)
    d = fpr[1:] - fpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.0).sum(-1)


def roc_curve(input, targ):
    "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
    targ = targ == 1
    desc_score_indices = torch.flip(input.argsort(-1), [-1])
    input = input[desc_score_indices]
    targ = targ[desc_score_indices]
    d = input[1:] - input[:-1]
    distinct_value_indices = torch.nonzero(d).transpose(0, 1)[0]
    threshold_idxs = torch.cat(
        (distinct_value_indices, torch.LongTensor([len(targ) - 1]).to(targ.device))
    )
    tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    if tps[0] != 0 or fps[0] != 0:
        zer = fps.new_zeros(1)
        fps = torch.cat((zer, fps))
        tps = torch.cat((zer, tps))
    fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]
    return fpr, tpr
