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
    lam_color: float = 1.0 #损失函数中的权重
    lam_shape: float = 1.0

    recon_loss: str = "bce_logits"  # "bce_logits" | "mse"
    grad_clip_norm: Optional[float] = None # 梯度裁剪阈值，目前为None

    use_amp: bool = False  #是否启用混合精度（AMP），GPU 上可加速/省显存
    log_every: int = 20 #每多少个 step 打印一次训练日志

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
    return kld_per.mean()

def generate_drop_path():
    # 随机生成 drop_path 类型，依据设定的比例
    rand_val = random.random()  # 生成一个 [0, 1) 范围内的随机数
    if rand_val < 0.7:
        return "latent-only"  # 50% 概率选择 latent+skip
    elif rand_val < 0.9:
        return "latent+skip"  # 25% 概率选择 latent-only
    else:
        return "skip-only"    # 25% 概率选择 skip-only

def compute_losses(
    images: torch.Tensor,
    x_logits: torch.Tensor,
    post: Dict[str, Any],
    cfg: TrainConfig,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss = recon + beta*KL
    """
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
    mu = post["mu"]
    lv = post["lv"]

    kl = kl_standard_normal(mu, lv)

    total = (
        recon
        + cfg.beta_kl * kl
    )

    logs = {
        "total": float(total.detach()),
        "recon": float(recon.detach()),
        "kl": float(kl.detach()*cfg.beta_kl),
    }
    return total, logs

def compute_losses_v2(
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
# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    ) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    agg = {}
    n = 0

    for step, batch in enumerate(loader):
        images, color_mh, shape_mh, count_oh, _img_fns = batch

        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
            out = model(images)

            # 兼容 forward 返回 (x_logits, post, heads) 或 dict
            if isinstance(out, (tuple, list)) and len(out) == 2:
                x_logits, post = out
            elif isinstance(out, dict):
                x_logits = out["x_logits"]
                post = out["post"]
            else:
                raise RuntimeError("Model forward output must be (x_logits, post, heads) or dict with keys.")

            loss, logs = compute_losses(images, x_logits, post, cfg)

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

    print(
            f"[train] epoch {epoch}"
                + " ".join([f"{k}={agg[k]:.4f}" for k in ("total", "recon", "kl")])
            )
    return agg

def train_one_epoch_v2(
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

            loss, logs = compute_losses_v2(images, x_logits, post, heads, color_mh, shape_mh, cfg, drop_path)

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
def evaluate(
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

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
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
                if best_val > train_logs["total"]:
                    save_checkpoint(model, optimizer, cfg, epoch, best_val, ckpt_path)
                    best_val = train_logs["total"]

    return {"best_val": best_val, "history": history, "ckpt_path": ckpt_path}


def fit_v2(
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
