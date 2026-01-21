
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入 nn 模块

from typing import Optional, Dict, Any  # 类型标注（可选）

def loss_color(  # 一个函数：loss + 每步严格对齐的对/错 count
    logits_seq: torch.Tensor,            # (B, Step, 8) 每一步 logits
    targets: torch.Tensor,               # (B, 8) multi-hot 标签（0/1）
    mode: str = "last",                  # "last" | "all" | "weighted" | "last+aux"
    threshold: float = 0.5,              # sigmoid 后阈值化
    step_weights: torch.Tensor = None,   # (Step,) weighted 模式可选
    aux_weight: float = 0.1,             # last+aux 模式可选
    pos_weight: torch.Tensor = None,     # (8,) BCE pos_weight 可选
):
    B, S, C = logits_seq.shape  # 解析维度
    targets = targets.float()  # BCEWithLogitsLoss 需要 float 标签

    # ---------- 1) 计算 loss（可切换） ----------
    loss_fn = nn.BCEWithLogitsLoss(  # BCEWithLogits = sigmoid + BCE
        reduction="mean",  # 输出标量
        pos_weight=pos_weight.to(logits_seq.device) if pos_weight is not None else None  # 可选：类别不均衡
    )

    if mode == "last":  # 只监督最后一步
        loss = loss_fn(logits_seq[:, -1, :], targets)  # last-step loss
    elif mode == "all":  # 每一步都监督，取平均
        losses = []  # 存每步 loss
        for t in range(S):  # 遍历 step
            losses.append(loss_fn(logits_seq[:, t, :], targets))  # 计算当前步 loss
        loss = torch.stack(losses, dim=0).mean()  # 平均成标量
    elif mode == "weighted":  # 每一步监督，按权重加权
        losses = []  # 存每步 loss
        for t in range(S):  # 遍历 step
            losses.append(loss_fn(logits_seq[:, t, :], targets))  # 当前步 loss
        losses = torch.stack(losses, dim=0)  # (S,) 每步一个 loss
        if step_weights is None:  # 如果没给权重
            step_weights = torch.linspace(1.0, 2.0, steps=S, device=logits_seq.device)  # 默认后期更重
        else:
            step_weights = step_weights.to(logits_seq.device)  # 搬到 device
            assert step_weights.shape == (S,), f"step_weights 必须是 (Step,)={S}"  # 检查形状
        step_weights = step_weights / step_weights.sum()  # 归一化
        loss = (losses * step_weights).sum()  # 加权求和
    elif mode == "last+aux":  # 最后一步主导 + 早期步辅助
        main_loss = loss_fn(logits_seq[:, -1, :], targets)  # 主损失：最后一步
        if S == 1:  # 只有一步就直接返回
            loss = main_loss  # 退化为 last
        else:
            aux_losses = []  # 存早期步 loss
            for t in range(S - 1):  # 只遍历早期步
                aux_losses.append(loss_fn(logits_seq[:, t, :], targets))  # 早期步 loss
            aux_loss = torch.stack(aux_losses, dim=0).mean()  # 早期步平均
            loss = main_loss + aux_weight * aux_loss  # 合成总 loss
    else:
        raise ValueError("mode 必须是 'last' | 'all' | 'weighted' | 'last+aux'")  # 报错

    # ---------- 2) 计算每一步 strict correct/wrong counts ----------
    with torch.no_grad():  # 统计不需要梯度
        probs = torch.sigmoid(logits_seq)  # (B, S, 8) 转概率（仅用于阈值化）
        preds = (probs >= threshold).float()  # (B, S, 8) 阈值化得到 0/1

        correct_counts = torch.zeros(S, device=logits_seq.device)  # (S,) 每步全对数量
        wrong_counts = torch.zeros(S, device=logits_seq.device)  # (S,) 每步非全对数量

        for t in range(S):  # 遍历每一步
            eq = (preds[:, t, :] == targets)  # (B, 8) 每一位是否正确
            all_ok = eq.all(dim=1)  # (B,) 每张图是否 8 位全对
            n_ok = all_ok.sum()  # 全对数量
            correct_counts[t] = n_ok  # 记录 correct
            wrong_counts[t] = B - n_ok  # 记录 wrong
 
    return loss, correct_counts, wrong_counts  # 返回：loss + 每步 n_correct/n_wrong（用于 epoch 累加）

def loss_count(  # one-hot count 的 loss + 每步对/错统计
    logits_seq: torch.Tensor,            # (B, Step, 11) 每一步 logits
    targets_oh: torch.Tensor,            # (B, 11) one-hot 标签（0..10）
    mode: str = "last",                  # "last" | "all" | "weighted" | "last+aux"
    step_weights: Optional[torch.Tensor] = None,  # (Step,) weighted 模式可选
    aux_weight: float = 0.1,             # last+aux 模式可选
):
    B, S, K = logits_seq.shape  # B=batch, S=step, K=11 类

    # --- one-hot -> class index (B,) ---
    # 假设 targets_oh 每行恰好一个 1（否则 argmax 仍会给一个 index，但语义不严谨）
    targets_idx = targets_oh.argmax(dim=1).long()  # (B,)

    # ---------- 1) 计算 loss（可切换） ----------
    loss_fn = nn.CrossEntropyLoss(reduction="mean")  # CE 输入 logits, target=类别index

    if mode == "last":  # 只监督最后一步
        loss = loss_fn(logits_seq[:, -1, :], targets_idx)  # last-step CE
    elif mode == "all":  # 每一步都监督，取平均
        losses = []
        for t in range(S):
            losses.append(loss_fn(logits_seq[:, t, :], targets_idx))
        loss = torch.stack(losses, dim=0).mean()
    elif mode == "weighted":  # 每一步监督，按权重加权
        losses = []
        for t in range(S):
            losses.append(loss_fn(logits_seq[:, t, :], targets_idx))
        losses = torch.stack(losses, dim=0)  # (S,)
        if step_weights is None:
            step_weights = torch.linspace(1.0, 2.0, steps=S, device=logits_seq.device)
        else:
            step_weights = step_weights.to(logits_seq.device)
            assert step_weights.shape == (S,), f"step_weights 必须是 (Step,)={S}"
        step_weights = step_weights / step_weights.sum()
        loss = (losses * step_weights).sum()
    elif mode == "last+aux":  # 最后一步主导 + 早期步辅助
        main_loss = loss_fn(logits_seq[:, -1, :], targets_idx)
        if S == 1:
            loss = main_loss
        else:
            aux_losses = []
            for t in range(S - 1):
                aux_losses.append(loss_fn(logits_seq[:, t, :], targets_idx))
            aux_loss = torch.stack(aux_losses, dim=0).mean()
            loss = main_loss + aux_weight * aux_loss
    else:
        raise ValueError("mode 必须是 'last' | 'all' | 'weighted' | 'last+aux'")

    # ---------- 2) 每一步 strict correct/wrong counts ----------
    with torch.no_grad():
        pred_idx = logits_seq.argmax(dim=2)  # (B, S) 每一步预测类别
        correct_counts = torch.zeros(S, device=logits_seq.device)  # (S,)
        wrong_counts = torch.zeros(S, device=logits_seq.device)    # (S,)

        for t in range(S):
            ok = (pred_idx[:, t] == targets_idx)  # (B,)
            n_ok = ok.sum()
            correct_counts[t] = n_ok
            wrong_counts[t] = B - n_ok

    return loss, correct_counts, wrong_counts

def loss_per_color_count(
    logits_seq: torch.Tensor,            # (B, Step, 8, 11)
    targets_oh: torch.Tensor,            # (B, 8, 11)
    mode: str = "last",                  # "last" | "all" | "weighted" | "last+aux"
    step_weights: Optional[torch.Tensor] = None,  # (Step,)
    aux_weight: float = 0.1,             # last+aux
    return_micro: bool = False,           # 是否额外返回 micro acc（按颜色位点） 按颜色粒度统计（总共 B*8 个颜色位点，预测对多少）
):
    B, S, K8, K11 = logits_seq.shape  # B=batch, S=step, K8=8 colors, K11=11 classes
    assert targets_oh.shape == (B, K8, K11), f"targets_oh 应为 (B,8,11)，但得到 {targets_oh.shape}"

    # --- one-hot -> class index: (B, 8) ---
    targets_idx = targets_oh.argmax(dim=2).long()  # 每个颜色的数量类别 0..10

    # ---------- 1) loss（可切换） ----------
    # CE 输入：(N, C) logits + (N,) target_index
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def step_ce(t: int) -> torch.Tensor:
        # logits_t: (B, 8, 11) -> reshape 为 (B*8, 11)
        logits_t = logits_seq[:, t, :, :].reshape(B * K8, K11)
        targets_t = targets_idx.reshape(B * K8)
        return loss_fn(logits_t, targets_t)

    if mode == "last":
        loss = step_ce(S - 1)
    elif mode == "all":
        losses = [step_ce(t) for t in range(S)]
        loss = torch.stack(losses, dim=0).mean()
    elif mode == "weighted":
        losses = [step_ce(t) for t in range(S)]
        losses = torch.stack(losses, dim=0)  # (S,)
        if step_weights is None:
            step_weights = torch.linspace(1.0, 2.0, steps=S, device=logits_seq.device)
        else:
            step_weights = step_weights.to(logits_seq.device)
            assert step_weights.shape == (S,), f"step_weights 必须是 (Step,)={S}"
        step_weights = step_weights / step_weights.sum()
        loss = (losses * step_weights).sum()
    elif mode == "last+aux":
        main_loss = step_ce(S - 1)
        if S == 1:
            loss = main_loss
        else:
            aux_losses = [step_ce(t) for t in range(S - 1)]
            aux_loss = torch.stack(aux_losses, dim=0).mean()
            loss = main_loss + aux_weight * aux_loss
    else:
        raise ValueError("mode 必须是 'last' | 'all' | 'weighted' | 'last+aux'")

    # ---------- 2) 统计每一步 strict correct/wrong（按 batch 图片为单位） ----------
    with torch.no_grad():
        # pred_idx: (B, S, 8)
        pred_idx = logits_seq.argmax(dim=3)  # 对 11 类取 argmax
        correct_counts = torch.zeros(S, device=logits_seq.device)
        wrong_counts = torch.zeros(S, device=logits_seq.device)

        micro_acc = torch.zeros(S, device=logits_seq.device) if return_micro else None

        for t in range(S):
            # per_color_ok: (B, 8)
            per_color_ok = (pred_idx[:, t, :] == targets_idx)

            # strict：一张图 8 个颜色全对才算对 -> (B,)
            all_ok = per_color_ok.all(dim=1)
            n_ok = all_ok.sum()
            correct_counts[t] = n_ok
            wrong_counts[t] = B - n_ok

            # micro：按颜色位点统计正确率（B*8 个位置）
            if return_micro:
                micro_acc[t] = per_color_ok.float().mean()

    return loss, correct_counts, wrong_counts, (micro_acc.detach().cpu() if return_micro else None)

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    loss_mode: str = "last",
    threshold: float = 0.5,
    step_weights: Optional[torch.Tensor] = None,
    aux_weight: float = 0.1,
    pos_weight: Optional[torch.Tensor] = None,   # 只给 BCE 的 color presence 用
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, Any]:
    model.train()

    running_loss = 0.0
    num_batches = 0

    # ✅ 三套统计各自初始化
    correct_sum_color = torch.zeros(steps, device=device)
    wrong_sum_color   = torch.zeros(steps, device=device)

    correct_sum_count = torch.zeros(steps, device=device)
    wrong_sum_count   = torch.zeros(steps, device=device)

    correct_sum_color_count = torch.zeros(steps, device=device)
    wrong_sum_color_count   = torch.zeros(steps, device=device)

    for step, batch in enumerate(dataloader):
        images, color_mh, shape_mh, count_oh, color_count, shape_count = batch

        images      = images.to(device)
        # color_mh    = color_mh.to(device)
        count_oh    = count_oh.to(device)
        # color_count = color_count.to(device)

        # ✅ model 应该返回 3 个 logits 序列
        logits_counts = model(images, steps=steps)

        # ===== loss 1: color presence (BCE) =====
        # lc, corr_c, wrong_c = loss_color(
        #     logits_seq=logits_colors,      # (B,S,8)
        #     targets=color_mh,              # (B,8)
        #     mode=loss_mode,
        #     threshold=threshold,
        #     step_weights=step_weights,
        #     aux_weight=aux_weight,
        #     pos_weight=pos_weight,         # ✅ 只有这里需要
        # )

        # ===== loss 2: total count (CE) =====
        lcnt, corr_cnt, wrong_cnt = loss_count(
            logits_seq=logits_counts,      # (B,S,11)
            targets_oh=count_oh,           # (B,11)
            mode=loss_mode,
            step_weights=step_weights,
            aux_weight=aux_weight,
        )

        # ===== loss 3: per-color count (CE) =====
        # 注意：我之前函数名叫 loss_per_color_count，targets_oh shape=(B,8,11)
        # lpc, corr_pc, wrong_pc, _micro = loss_per_color_count(
        #     logits_seq=logits_color_count, # (B,S,8,11)
        #     targets_oh=color_count,        # (B,8,11)
        #     mode=loss_mode,
        #     step_weights=step_weights,
        #     aux_weight=aux_weight,
        #     return_micro=False,            # 先不统计 micro 也行
        # )

        # loss_total = lc + lcnt + lpc
        loss_total = lcnt

        optimizer.zero_grad()
        loss_total.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        running_loss += float(loss_total.item())
        num_batches += 1

        # ✅ 累加统计
        # correct_sum_color += corr_c.detach()
        # wrong_sum_color   += wrong_c.detach()

        correct_sum_count += corr_cnt.detach()
        wrong_sum_count   += wrong_cnt.detach()

        # correct_sum_color_count += corr_pc.detach()
        # wrong_sum_color_count   += wrong_pc.detach()

    avg_loss = running_loss / max(num_batches, 1)

    # ✅ 每个任务各自算 acc
    # total_color = correct_sum_color + wrong_sum_color
    # acc_color = correct_sum_color / torch.clamp(total_color, min=1.0)

    total_count = correct_sum_count + wrong_sum_count
    acc_count = correct_sum_count / torch.clamp(total_count, min=1.0)

    # total_color_count = correct_sum_color_count + wrong_sum_color_count
    # acc_color_count = correct_sum_color_count / torch.clamp(total_color_count, min=1.0)

    return {
        "loss": avg_loss,
        # "acc_per_step_color": acc_color.detach().cpu(),
        "acc_per_step_count": acc_count.detach().cpu(),
        # "acc_per_step_color_count": acc_color_count.detach().cpu(),
    }


@torch.no_grad()  # 验证不需要梯度
def evaluate_one_epoch(  # 评估一个 epoch（只统计，不反传）
    model: torch.nn.Module,  # 模型
    dataloader,  # 验证 DataLoader
    device: torch.device,  # 设备
    steps: int,  # recurrent steps
    threshold: float = 0.5,  # strict 阈值
) -> Dict[str, Any]:
    model.eval()  # 进入评估模式

    correct_sum = torch.zeros(steps, device=device)  # (steps,) 累加全对
    wrong_sum = torch.zeros(steps, device=device)  # (steps,) 累加全错

    for step, batch in enumerate(dataloader):
        images, color_mh, shape_mh, count_oh, _img_fns = batch
        x = images.to(device)  # 输入搬到 device
        y = color_mh.to(device)  # 标签搬到 device（loss 内部会 float）

        logits_seq = model(x, steps=steps)  # (B, steps, 8)

        # 只做 strict count，不算 loss（你现在也不需要）
        _, correct_counts, wrong_counts = loss(  # 复用 loss 函数的统计部分
            logits_seq=logits_seq,  # logits
            targets=y,  # 标签
            mode="last",  # 这里 mode 对统计没影响（只影响 loss），随便给一个
            threshold=threshold,  # 阈值
        )

        correct_sum += correct_counts.detach()  # 累加
        wrong_sum += wrong_counts.detach()  # 累加

    total_per_step = correct_sum + wrong_sum  # 总数
    acc_per_step = correct_sum / torch.clamp(total_per_step, min=1.0)  # strict accuracy

    return {  # 返回评估统计
        "correct_per_step": correct_sum.detach().cpu(),
        "wrong_per_step": wrong_sum.detach().cpu(),
        "acc_per_step": acc_per_step.detach().cpu(),
        "num_samples_per_step": total_per_step.detach().cpu(),
    }


def train_loop(  # 完整训练循环（多 epoch）
    model: torch.nn.Module,  # 模型
    train_loader,  # 训练 DataLoader
    device: torch.device,  # 设备
    steps: int,  # recurrent steps
    epochs: int = 20,  # epoch 数
    lr: float = 1e-3,  # 学习率
    loss_mode: str = "last",  # 损失模式
    threshold: float = 0.5,  # strict 阈值
    step_weights: Optional[torch.Tensor] = None,  # weighted 模式用
    aux_weight: float = 0.1,  # last+aux 用
    pos_weight: Optional[torch.Tensor] = None,  # 类别不均衡
    grad_clip_norm: Optional[float] = 1.0,  # 梯度裁剪
    val_loader=None,  # 验证 DataLoader（可选）
):
    model = model.to(device)  # 模型搬到 device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam 优化器

    history = {  # 用 dict 存训练过程（后面画图用）
        "train_loss": [],  # 每 epoch 平均 loss
        "train_acc_per_step": [],  # 每 epoch 每步 accuracy
        "val_acc_per_step": [],  # 每 epoch 每步 val accuracy（可选）
    }

    for epoch in range(1, epochs + 1):  # 遍历 epoch
        train_stats = train_one_epoch(  # 训练一个 epoch
            model=model,  # 模型
            dataloader=train_loader,  # train loader
            optimizer=optimizer,  # 优化器
            device=device,  # device
            steps=steps,  # steps
            loss_mode=loss_mode,  # loss 模式
            threshold=threshold,  # 阈值
            step_weights=step_weights,  # step 权重（可选）
            aux_weight=aux_weight,  # aux 权重
            pos_weight=pos_weight,  # pos_weight（可选）
            grad_clip_norm=grad_clip_norm,  # 裁剪
        )

        history["train_loss"].append(train_stats["loss"])  # 记录 loss
        history["train_acc_per_step"].append(train_stats["acc_per_step_count"])  # 记录每步 acc

        # 打印训练摘要：只打印最后一步的 acc（你也可以打印整条曲线）
        last_acc = float(train_stats["acc_per_step_count"][-1])  # 取最后一步 accuracy
        print(f"Epoch {epoch:03d} | loss={train_stats['loss']:.4f} | train_acc(t=end)={last_acc:.4f}")  # 打印

        if val_loader is not None:  # 如果有验证集
            val_stats = evaluate_one_epoch(  # 评估
                model=model,  # 模型
                dataloader=val_loader,  # val loader
                device=device,  # device
                steps=steps,  # steps
                threshold=threshold,  # 阈值
            )
            history["val_acc_per_step"].append(val_stats["acc_per_step"])  # 记录验证 acc
            val_last_acc = float(val_stats["acc_per_step"][-1])  # 最后一步 val acc
            print(f"         | val_acc(t=end)={val_last_acc:.4f}")  # 打印验证摘要

            # acc = val_stats["acc_per_step"]  # (steps,)
            # for t, a in enumerate(acc.tolist(), start=1):
            #     print(f"val step {t}: acc={a:.4f}")
    return history  # 返回历史记录（之后做可视化）
