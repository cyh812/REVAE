import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import random


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    beta_kl: float = 1.0
    weight_decay: float = 0.0
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
class TDVAECfg:
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

def generate_drop_path():
    # 随机生成 drop_path 类型，依据设定的比例
    rand_val = random.random()  # 生成一个 [0, 1) 范围内的随机数
    if rand_val < 0.7:
        return "latent-only"  # 50% 概率选择 latent+skip
    elif rand_val < 0.9:
        return "latent+skip"  # 25% 概率选择 latent-only
    else:
        return "skip-only"    # 25% 概率选择 skip-only

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

def mlr_losses(
    images: torch.Tensor,
    x_logits: torch.Tensor,
    post: Dict[str, Any],
    heads: Dict[str, Any],
    color_mh: torch.Tensor,
    shape_mh: torch.Tensor,
    cfg: TrainConfig,
    drop_path: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

    B = images.size(0)

    # 1) reconstruction
    if cfg.recon_loss == "bce_logits":
        # images must be in [0,1]
        recon = F.binary_cross_entropy_with_logits(x_logits, images, reduction="sum") / B
    elif cfg.recon_loss == "mse":
        recon = F.mse_loss(torch.sigmoid(x_logits), images, reduction="mean")
    elif cfg.recon_loss == "l1":
        recon = F.l1_loss(torch.sigmoid(x_logits), images, reduction="mean")
    else:
        raise ValueError(f"Unknown recon_loss: {cfg.recon_loss}")

    # 2) KL per group
    mu_c = post["mu_color"]
    lv_c = post["lv_color"]
    mu_s = post["mu_shape"]
    lv_s = post["lv_shape"]

    kl_color = kl_standard_normal(mu_c, lv_c)
    kl_shape = kl_standard_normal(mu_s, lv_s)
    kl = kl_color + kl_shape

    if drop_path != "skip-only":
        # 3) supervised heads
        color_logits = heads["color_logits"]
        shape_logits = heads["shape_logits"]

        loss_color = F.binary_cross_entropy_with_logits(color_logits, color_mh, reduction="sum") / B 
        loss_shape = F.binary_cross_entropy_with_logits(shape_logits, shape_mh, reduction="sum") / B

        total = (
            recon
            + cfg.beta_kl * kl
            + cfg.lam_color * loss_color
            + cfg.lam_shape * loss_shape
        )

        logs = {
            "total": float(total.detach()),
            "recon": float(recon.detach()),
            "kl": float(kl.detach()*cfg.beta_kl),
            "loss": float(loss_color.detach()*cfg.lam_color) + float(loss_shape.detach()*cfg.lam_shape)
        }
    
    else:
        total = (
            recon
        )

        logs = {
            "total": float(total.detach()),
            "recon": float(recon.detach()),
            "kl": 0.0,
            "loss": 0.0
        }
    return total, logs

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
def train_one_epoch_tdvae(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TDVAECfg,
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
def evaluate_tdvae(
    model: torch.nn.Module,
    val_dl: torch.utils.data.DataLoader,
    cfg: TDVAECfg,
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
    cfg,
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

def fit_tdvae(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: Optional[torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    cfg: TDVAECfg,
    ) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)
    device = torch.device(cfg.device)
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    history = {"train": [], "val": []}
    best_metric = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_logs = train_one_epoch_tdvae(model, train_dl, optimizer, cfg)
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



def train_one_epoch_mlr(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    ) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.use_amp and device.type == "cuda"))

    agg = {}
    n = 0

    for step, batch in enumerate(loader):
        images, color_mh, shape_mh, count_oh, _img_fns = batch
        drop_path = generate_drop_path()
        images = images.to(device, non_blocking=True)
        color_mh = color_mh.to(device, non_blocking=True).float()
        shape_mh = shape_mh.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(cfg.use_amp and device.type == "cuda")):
            x_logits, heads, post = model(images, drop_path)

            loss, logs = mlr_losses(images, x_logits, post, heads, color_mh, shape_mh, cfg, drop_path)

        scaler.scale(loss).backward()

        if cfg.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        # aggregate logs
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1

    # mean logs
    for k in agg:
        agg[k] /= max(n, 1)

    if n > 0:
        save_epoch_samples_from_last_batch(
            model=model,
            images=images,          # 这里就是最后一个batch的images（已在device上）
            device=device,
            epoch=epoch,
            out_dir="sample",
            num_samples=5,
        )

    print(
            f"[train] epoch {epoch}: "
                + " ".join([f"{k}={agg[k]:.4f}" for k in ("total", "recon", "kl", "loss")])
            )
    return agg


@torch.no_grad()
def evaluate_mlr(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: TrainConfig,
    split_name: str = "val",
    ) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)

    agg = {}
    n = 0

    for batch in loader:
        images, color_mh, shape_mh, count_oh, _img_fns = batch

        images = images.to(device, non_blocking=True)
        color_mh = color_mh.to(device, non_blocking=True).float()
        shape_mh = shape_mh.to(device, non_blocking=True).float()

        x_logits, heads, post = model(images)

        loss, logs = compute_losses_v2(images, x_logits, post, heads, color_mh, shape_mh, cfg, drop_path="latent+skip")

        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1

    for k in agg:
        agg[k] /= max(n, 1)
        # epoch结束：用最后一个batch可视化并保存

    print(
            f"[train] epoch {epoch}: "
                + " ".join([f"{k}={agg[k]:.4f}" for k in ("total", "recon", "kl", "loss")])
            )
    return agg


def fit_mlr(
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
        train_logs = train_one_epoch_v2(model, train_loader, optimizer, cfg, epoch)
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
                if best_val > train_logs["total"]:
                    save_checkpoint(model, optimizer, cfg, epoch, best_val, ckpt_path)
                    best_val = train_logs["total"]

    return {"best_val": best_val, "history": history, "ckpt_path": ckpt_path}

def save_epoch_samples_from_last_batch(
    model: torch.nn.Module,
    images: torch.Tensor,   # (B,3,224,224) already on device
    device: torch.device,
    epoch: int,
    out_dir: str = "sample",
    num_samples: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)

    # 选最多 num_samples 张
    B = images.size(0)
    k = min(num_samples, B)
    idx = torch.randperm(B, device=images.device)[:k]
    x = images[idx]  # (k,3,H,W)

    modes = ["latent+skip", "latent-only", "color-only", "shape-only", "skip-only"]
    col_titles = ["Input", "latent+skip", "latent-only", "color-only", "shape-only", "skip-only"]

    model_was_training = model.training
    model.eval()

    # 先准备所有结果：第0列是输入，其余列是5种重建
    with torch.no_grad():
        outs = {}
        for m in modes:
            x_logits, _heads = model.predict(x, pre_type=m)
            outs[m] = torch.sigmoid(x_logits).clamp(0, 1).detach().cpu()  # (k,3,H,W)

    x_cpu = x.detach().cpu()

    # 画 k x 6
    nrows, ncols = k, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.0))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    def to_img(t):  # (3,H,W) -> (H,W,3)
        return t.permute(1, 2, 0).numpy()

    for r in range(nrows):
        # Input
        axes[r, 0].imshow(to_img(x_cpu[r]))
        axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title(col_titles[0])

        # 5 recon columns
        for c, m in enumerate(modes, start=1):
            axes[r, c].imshow(to_img(outs[m][r]))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c])

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 恢复训练状态
    if model_was_training:
        model.train()
