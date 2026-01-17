import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    beta_kl: float = 1.0

    lam_color: float = 1.0 #损失函数中的权重
    lam_shape: float = 1.0
    lam_count: float = 1.0

    recon_loss: str = "bce_logits"  # "bce_logits" | "mse"
    grad_clip_norm: Optional[float] = None # 梯度裁剪阈值，目前为None

    use_amp: bool = False  #是否启用混合精度（AMP），GPU 上可加速/省显存
    log_every: int = 20 #每多少个 step 打印一次训练日志

    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "revae.pt"
    save_best: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainCfg:
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 0.0
    beta_y: float = 1.0
    beta_z: float = 1.0
    recon_type: str = "bce_logits"   # or "mse"
    img_size: int = 64              # TDVAE currently assumes 64x64
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = True
    log_every: int = 50
    sample_from: str = "posterior"  # "posterior" or "prior"

    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "revae.pt"
    save_best: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, var) || N(0, I) ) for diagonal Gaussian.
    Returns scalar mean over batch (sum over dims, mean over batch).
    """
    # per-sample sum over dims
    kld_per = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
    return kld_per

def kl_prior_normal(mu_q:torch.Tensor, logvar_q:torch.Tensor, mu_p:torch.Tensor, logvar_p:torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu_q, exp(logvar_q)) || N(mu_p, exp(logvar_p)) )
    """
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kld_per = 0.5 * torch.sum(
        (logvar_p - logvar_q) + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0,
        dim=1
    )
    return kld_per

def tdvae_loss(outputs, x, beta_y=1.0, beta_z=1.0, recon_type="bce_logits"):
    """
    outputs: dict from model.forward()
    x: [B,3,64,64] in [0,1] if bce_logits
    """
    x_logits = outputs["x_logits"]

    if recon_type == "bce_logits":
        # sum over pixels/channels, then mean over batch
        rec = F.binary_cross_entropy_with_logits(
            x_logits, x, reduction="none"
        ).flatten(1).sum(dim=1)  # [B]
    elif recon_type == "mse":
        x_hat = torch.sigmoid(x_logits)
        rec = F.mse_loss(x_hat, x, reduction="none").flatten(1).sum(dim=1)  # [B]
    else:
        raise ValueError("recon_type must be 'bernoulli' or 'mse'")

    kl_y = kl_standard_normal(outputs["mu_y"], outputs["logvar_y"])  # [B]
    kl_z = kl_prior_normal(
        outputs["mu_z_post"], outputs["logvar_z_post"],
        outputs["mu_z_prior"], outputs["logvar_z_prior"]
    )  # [B]

    loss = (rec + beta_y * kl_y + beta_z * kl_z).mean()

    return {
        "loss": loss,
        "rec": rec.mean(),
        "kl_y": kl_y.mean(),
        "kl_z": kl_z.mean(),
    }

# def compute_losses(
#     images: torch.Tensor,
#     x_logits: torch.Tensor,
#     post: Dict[str, Any],
#     heads: Dict[str, Any],
#     color_mh: torch.Tensor,
#     shape_mh: torch.Tensor,
#     count_oh: torch.Tensor,
#     cfg: TrainConfig,
# ) -> Tuple[torch.Tensor, Dict[str, float]]:
#     """
#     Total loss = recon + beta*KL + supervised heads losses
#     """
#     B = images.size(0)

#     # 1) reconstruction
#     if cfg.recon_loss == "bce_logits":
#         # images must be in [0,1]
#         recon = F.binary_cross_entropy_with_logits(x_logits, images, reduction="sum") / B
#     elif cfg.recon_loss == "mse":
#         recon = F.mse_loss(torch.sigmoid(x_logits), images, reduction="mean")
#     elif cfg.recon_loss == "l1":
#         recon = F.l1_loss(torch.sigmoid(x_logits), images, reduction="mean")
#     else:
#         raise ValueError(f"Unknown recon_loss: {cfg.recon_loss}")

#     # 2) KL per group
#     mu_c = post["mu_color"]
#     lv_c = post["lv_color"]
#     mu_s = post["mu_shape"]
#     lv_s = post["lv_shape"]
#     mu_n = post["mu_count"]
#     lv_n = post["lv_count"]

#     kl_c = kl_standard_normal(mu_c, lv_c)
#     kl_s = kl_standard_normal(mu_s, lv_s)
#     kl_n = kl_standard_normal(mu_n, lv_n)
#     kl = (kl_c + kl_s + kl_n) / 3.0

#     # 3) supervised heads
#     color_logits = heads["color_logits"]
#     shape_logits = heads["shape_logits"]
#     count_logits = heads["count_logits"]

#     loss_color = F.binary_cross_entropy_with_logits(color_logits, color_mh, reduction="mean")
#     loss_shape = F.binary_cross_entropy_with_logits(shape_logits, shape_mh, reduction="mean")

#     count_cls = count_oh.argmax(dim=1)  # (B,)
#     loss_count = F.cross_entropy(count_logits, count_cls, reduction="mean")

#     total = (
#         recon
#         + cfg.beta_kl * kl
#         + cfg.lam_color * loss_color
#         + cfg.lam_shape * loss_shape
#         + cfg.lam_count * loss_count
#     )

#     logs = {
#         "total": float(total.detach()),
#         "recon": float(recon.detach()),
#         "kl": float(kl.detach()*cfg.beta_kl),
#         "kl_color": float(kl_c.detach()),
#         "kl_shape": float(kl_s.detach()),
#         "kl_count": float(kl_n.detach()),
#         "loss_color": float(loss_color.detach()*cfg.lam_color),
#         "loss_shape": float(loss_shape.detach()*cfg.lam_shape),
#         "loss_count": float(loss_count.detach()*cfg.lam_count),
#     }
#     return total, logs


# # ----------------------------
# # Train / Eval
# # ----------------------------
# def train_one_epoch(
#     model: torch.nn.Module,
#     loader: torch.utils.data.DataLoader,
#     optimizer: torch.optim.Optimizer,
#     cfg: TrainConfig,
#     epoch: int,
# ) -> Dict[str, float]:
#     model.train()
#     device = torch.device(cfg.device)

#     scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

#     agg = {}
#     n = 0

#     for step, batch in enumerate(loader):
#         images, color_mh, shape_mh, count_oh, _img_fns = batch

#         images = images.to(device, non_blocking=True)
#         color_mh = color_mh.to(device, non_blocking=True).float()
#         shape_mh = shape_mh.to(device, non_blocking=True).float()
#         count_oh = count_oh.to(device, non_blocking=True).float()

#         optimizer.zero_grad(set_to_none=True)

#         with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
#             out = model(images)

#             # 兼容 forward 返回 (x_logits, post, heads) 或 dict
#             if isinstance(out, (tuple, list)) and len(out) == 3:
#                 x_logits, post, heads = out
#             elif isinstance(out, dict):
#                 x_logits = out["x_logits"]
#                 post = out["post"]
#                 heads = out["heads"]
#             else:
#                 raise RuntimeError("Model forward output must be (x_logits, post, heads) or dict with keys.")

#             loss, logs = compute_losses(images, x_logits, post, heads, color_mh, shape_mh, count_oh, cfg)

#         scaler.scale(loss).backward()

#         if cfg.grad_clip_norm is not None:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

#         scaler.step(optimizer)
#         scaler.update()

#         # aggregate logs
#         for k, v in logs.items():
#             agg[k] = agg.get(k, 0.0) + v
#         n += 1

#         # if cfg.log_every and (step % cfg.log_every == 0):
#         #     print(
#         #         f"[train] epoch {epoch} step {step}/{len(loader)} "
#         #         + " ".join([f"{k}={logs[k]:.4f}" for k in ("total", "recon", "kl", "loss_color", "loss_shape", "loss_count")])
#         #     )

#     # mean logs
#     for k in agg:
#         agg[k] /= max(n, 1)

#     print(
#             f"[train] epoch {epoch}"
#                 + " ".join([f"{k}={agg[k]:.4f}" for k in ("total", "recon", "kl", "loss_color", "loss_shape", "loss_count")])
#             )
#     return agg


# @torch.no_grad()
# def evaluate(
#     model: torch.nn.Module,
#     loader: torch.utils.data.DataLoader,
#     cfg: TrainConfig,
#     split_name: str = "val",
# ) -> Dict[str, float]:
#     model.eval()
#     device = torch.device(cfg.device)

#     agg = {}
#     n = 0

#     for batch in loader:
#         images, color_mh, shape_mh, count_oh, _img_fns = batch

#         images = images.to(device, non_blocking=True)
#         color_mh = color_mh.to(device, non_blocking=True).float()
#         shape_mh = shape_mh.to(device, non_blocking=True).float()
#         count_oh = count_oh.to(device, non_blocking=True).float()

#         out = model(images)
#         if isinstance(out, (tuple, list)) and len(out) == 3:
#             x_logits, post, heads = out
#         elif isinstance(out, dict):
#             x_logits = out["x_logits"]
#             post = out["post"]
#             heads = out["heads"]
#         else:
#             raise RuntimeError("Model forward output must be (x_logits, post, heads) or dict with keys.")

#         loss, logs = compute_losses(images, x_logits, post, heads, color_mh, shape_mh, count_oh, cfg)

#         for k, v in logs.items():
#             agg[k] = agg.get(k, 0.0) + v
#         n += 1

#     for k in agg:
#         agg[k] /= max(n, 1)

#     print(
#         f"[{split_name}] "
#         + " ".join([f"{k}={agg[k]:.4f}" for k in ("total", "recon", "kl", "loss_color", "loss_shape", "loss_count")])
#     )
#     return agg

@torch.no_grad()
def recon_metrics_from_logits(x_logits: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Simple reconstruction metrics for monitoring.
    """
    x_hat = torch.sigmoid(x_logits)
    mse = F.mse_loss(x_hat, x, reduction="mean")
    bce = F.binary_cross_entropy(x_hat.clamp(1e-6, 1-1e-6), x, reduction="mean")
    return {"mse": mse, "bce": bce}

# ----------------------------
# One epoch train
# ----------------------------
def train_one_epoch(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainCfg,
) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    # Running averages
    sums = {"loss": 0.0, "rec": 0.0, "kl_y": 0.0, "kl_z": 0.0, "mse": 0.0, "bce": 0.0}
    n_batches = 0

    for step, batch in enumerate(train_dl):
        # batch: images, color_mh, shape_mh, count_oh, img_fns
        images = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            outputs = model(images, sample_from=cfg.sample_from)
            losses = tdvae_loss(
                outputs, images,
                beta_y=cfg.beta_y, beta_z=cfg.beta_z,
                recon_type=cfg.recon_type
            )
            loss = losses["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics (detach)
        with torch.no_grad():
            mets = recon_metrics_from_logits(outputs["x_logits"], images)

        for k in ["loss", "rec", "kl_y", "kl_z"]:
            sums[k] += float(losses[k].detach().cpu())
        sums["mse"] += float(mets["mse"].cpu())
        sums["bce"] += float(mets["bce"].cpu())
        n_batches += 1

    avg = {k: v / max(1, n_batches) for k, v in sums.items()}
    print(" | ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
    return avg

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_dl: torch.utils.data.DataLoader,
    cfg: TrainCfg,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)

    sums = {
        "loss_post": 0.0, "rec_post": 0.0, "kl_y": 0.0, "kl_z": 0.0, "mse_post": 0.0, "bce_post": 0.0,
        "mse_prior": 0.0, "bce_prior": 0.0, "rec_prior_proxy": 0.0,
    }
    n_batches = 0

    for batch in val_dl:
        images = batch[0].to(device, non_blocking=True)
        
        # Posterior path (standard)
        out_post = model(images, sample_from="posterior")
        losses = tdvae_loss(out_post, images, beta_y=cfg.beta_y, beta_z=cfg.beta_z, recon_type=cfg.recon_type)
        mets_post = recon_metrics_from_logits(out_post["x_logits"], images)

        # Prior path (top-down only) — same decoder, just feed z_prior
        out_prior = model(images, sample_from="prior")
        mets_prior = recon_metrics_from_logits(out_prior["x_logits"], images)

        # "rec_prior_proxy": reconstruction error using prior z (not part of ELBO by default, but useful to track)
        if cfg.recon_type == "bce_logits":
            rec_prior_per = F.binary_cross_entropy_with_logits(out_prior["x_logits"], images, reduction="none").flatten(1).sum(dim=1)
        else:
            rec_prior_per = F.mse_loss(torch.sigmoid(out_prior["x_logits"]), images, reduction="none").flatten(1).sum(dim=1)

        sums["loss_post"] += float(losses["loss"].cpu())
        sums["rec_post"] += float(losses["rec"].cpu())
        sums["kl_y"] += float(losses["kl_y"].cpu())
        sums["kl_z"] += float(losses["kl_z"].cpu())
        sums["mse_post"] += float(mets_post["mse"].cpu())
        sums["bce_post"] += float(mets_post["bce"].cpu())

        sums["mse_prior"] += float(mets_prior["mse"].cpu())
        sums["bce_prior"] += float(mets_prior["bce"].cpu())
        sums["rec_prior_proxy"] += float(rec_prior_per.mean().cpu())

        n_batches += 1

    avg = {k: v / max(1, n_batches) for k, v in sums.items()}
    return avg

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainCfg,
    epoch: int,
    best_metric: float,
    path: str,
) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    map_location: str = "cpu",
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_metric", float("inf")))


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)

    best_val = float("inf")
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    history = {"train": [], "val": []}

    for epoch in range(1, cfg.epochs + 1):
        train_logs = train_one_epoch(model, train_loader, optimizer, cfg, epoch)
        history["train"].append(train_logs)

        if val_loader is not None:
            val_logs = evaluate(model, val_loader, cfg, split_name="val")
            history["val"].append(val_logs)

            # 以 val total 作为 best metric
            metric = val_logs["total"]
            if cfg.save_best and metric < best_val:
                best_val = metric
                save_checkpoint(model, optimizer, cfg, epoch, best_val, ckpt_path)
                print(f"[ckpt] saved best to {ckpt_path} (val_total={best_val:.4f})")
        else:
            # 没有 val，就每个 epoch 保存一次（可选）
            if not cfg.save_best:
                save_checkpoint(model, optimizer, cfg, epoch, best_val, ckpt_path)

    return {"best_val": best_val, "history": history, "ckpt_path": ckpt_path}


def fit_tdvae(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: Optional[torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    cfg: TrainCfg,
) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)
    device = torch.device(cfg.device)
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    history = {"train": [], "val": []}
    best_metric = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_logs = train_one_epoch(model, train_dl, optimizer, cfg)
        history["train"].append(train_logs)
        if best_metric > train_logs["loss"]:
            best_metric = train_logs["loss"]
            save_checkpoint(model, optimizer, cfg, epoch, best_metric, ckpt_path)
        if val_dl is not None:
            val_logs = evaluate(model, val_dl, device, cfg)
            print(
                f"[val]   loss_post {val_logs['loss_post']:.4f} rec_post {val_logs['rec_post']:.4f} "
                f"kl_y {val_logs['kl_y']:.4f} kl_z {val_logs['kl_z']:.4f} | "
                f"mse_post {val_logs['mse_post']:.6f} mse_prior {val_logs['mse_prior']:.6f} | "
                f"bce_post {val_logs['bce_post']:.6f} bce_prior {val_logs['bce_prior']:.6f}"
            )
            history["val"].append(val_logs)

    return history