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


def compute_losses(
    images: torch.Tensor,
    x_logits: torch.Tensor,
    post: Dict[str, Any],
    heads: Dict[str, Any],
    color_mh: torch.Tensor,
    shape_mh: torch.Tensor,
    count_oh: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss = recon + beta*KL + supervised heads losses
    """
    B = images.size(0)

    # 1) reconstruction
    if cfg.recon_loss == "bce_logits":
        # images must be in [0,1]
        recon = F.binary_cross_entropy_with_logits(x_logits, images, reduction="sum") / (B*images[0].numel())
    elif cfg.recon_loss == "mse":
        recon = F.mse_loss(torch.sigmoid(x_logits), images, reduction="mean") / (B*images[0].numel())
    else:
        raise ValueError(f"Unknown recon_loss: {cfg.recon_loss}")

    # 2) KL per group
    mu_c = post["mu_color"]
    lv_c = post["lv_color"]
    mu_s = post["mu_shape"]
    lv_s = post["lv_shape"]
    mu_n = post["mu_count"]
    lv_n = post["lv_count"]

    kl_c = kl_standard_normal(mu_c, lv_c)
    kl_s = kl_standard_normal(mu_s, lv_s)
    kl_n = kl_standard_normal(mu_n, lv_n)
    kl = kl_c + kl_s + kl_n

    # 3) supervised heads
    color_logits = heads["color_logits"]
    shape_logits = heads["shape_logits"]
    count_logits = heads["count_logits"]

    loss_color = F.binary_cross_entropy_with_logits(color_logits, color_mh, reduction="mean")
    loss_shape = F.binary_cross_entropy_with_logits(shape_logits, shape_mh, reduction="mean")

    count_cls = count_oh.argmax(dim=1)  # (B,)
    loss_count = F.cross_entropy(count_logits, count_cls, reduction="mean")

    total = (
        recon
        + cfg.beta_kl * kl
        + cfg.lam_color * loss_color
        + cfg.lam_shape * loss_shape
        + cfg.lam_count * loss_count
    )

    logs = {
        "total": float(total.detach()),
        "recon": float(recon.detach()),
        "kl": float(kl.detach()),
        "kl_color": float(kl_c.detach()),
        "kl_shape": float(kl_s.detach()),
        "kl_count": float(kl_n.detach()),
        "loss_color": float(loss_color.detach()),
        "loss_shape": float(loss_shape.detach()),
        "loss_count": float(loss_count.detach()),
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
        color_mh = color_mh.to(device, non_blocking=True).float()
        shape_mh = shape_mh.to(device, non_blocking=True).float()
        count_oh = count_oh.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
            out = model(images)

            # 兼容 forward 返回 (x_logits, post, heads) 或 dict
            if isinstance(out, (tuple, list)) and len(out) == 3:
                x_logits, post, heads = out
            elif isinstance(out, dict):
                x_logits = out["x_logits"]
                post = out["post"]
                heads = out["heads"]
            else:
                raise RuntimeError("Model forward output must be (x_logits, post, heads) or dict with keys.")

            loss, logs = compute_losses(images, x_logits, post, heads, color_mh, shape_mh, count_oh, cfg)

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

        if cfg.log_every and (step % cfg.log_every == 0):
            print(
                f"[train] epoch {epoch} step {step}/{len(loader)} "
                + " ".join([f"{k}={logs[k]:.4f}" for k in ("total", "recon", "kl", "loss_color", "loss_shape", "loss_count")])
            )

    # mean logs
    for k in agg:
        agg[k] /= max(n, 1)
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
        count_oh = count_oh.to(device, non_blocking=True).float()

        out = model(images)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            x_logits, post, heads = out
        elif isinstance(out, dict):
            x_logits = out["x_logits"]
            post = out["post"]
            heads = out["heads"]
        else:
            raise RuntimeError("Model forward output must be (x_logits, post, heads) or dict with keys.")

        loss, logs = compute_losses(images, x_logits, post, heads, color_mh, shape_mh, count_oh, cfg)

        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1

    for k in agg:
        agg[k] /= max(n, 1)

    print(
        f"[{split_name}] "
        + " ".join([f"{k}={agg[k]:.4f}" for k in ("total", "recon", "kl", "loss_color", "loss_shape", "loss_count")])
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
                save_checkpoint(model, optimizer, cfg, epoch, best_val, ckpt_path)

    return {"best_val": best_val, "history": history, "ckpt_path": ckpt_path}
