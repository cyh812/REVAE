import matplotlib.pyplot as plt  # 导入 matplotlib

def plot_history_epoch_curves(history: dict):
    """
    期望 history 至少包含：
      - history["train_loss"] : list[float]，长度=epochs
      - history["train_acc_per_step"] : list[tensor/array]，每个元素 shape=(steps,)
    可选：
      - history["val_acc_per_step"] : list[tensor/array]，每个元素 shape=(steps,)
      - history["val_loss"] : list[float]
    """

    # ---------- 1) 画 Loss vs Epoch ----------
    train_loss = history.get("train_loss", None)  # 取训练 loss 列表
    if train_loss is not None and len(train_loss) > 0:  # 如果有训练 loss
        epochs = list(range(1, len(train_loss) + 1))  # epoch 轴：1..E

        plt.figure()  # 新建一个图（不做子图）
        plt.plot(epochs, train_loss)  # 画训练 loss（默认颜色）
        plt.xlabel("epoch")  # x 轴
        plt.ylabel("loss")  # y 轴
        plt.title("Train Loss vs Epoch")  # 标题
        plt.show()  # 显示

    # val_loss = history.get("val_loss", None)  # 可选：验证 loss
    # if val_loss is not None and len(val_loss) > 0:  # 如果有验证 loss
    #     epochs = list(range(1, len(val_loss) + 1))  # epoch 轴
    #     plt.figure()  # 新图
    #     plt.plot(epochs, val_loss)  # 画验证 loss
    #     plt.xlabel("epoch")  # x 轴
    #     plt.ylabel("loss")  # y 轴
    #     plt.title("Val Loss vs Epoch")  # 标题
    #     plt.show()  # 显示

    # ---------- 2) 画 Acc vs Epoch（每一步一条线） ----------
    train_acc_seq = history.get("train_acc_per_step", None)  # 取训练 acc 序列（list）
    if train_acc_seq is not None and len(train_acc_seq) > 0:  # 如果存在
        E = len(train_acc_seq)  # epoch 数
        epochs = list(range(1, E + 1))  # epoch 轴

        # 把 list[steps] 变成二维列表 acc_matrix: shape=(E, steps)
        acc_matrix = []  # 保存每个 epoch 的 acc 向量
        for e in range(E):  # 遍历 epoch
            acc_vec = train_acc_seq[e]  # 当前 epoch 的 acc_per_step（tensor/np）
            acc_vec = acc_vec.detach().cpu().numpy() if hasattr(acc_vec, "detach") else acc_vec  # 转 numpy（兼容 tensor）
            acc_matrix.append(acc_vec)  # 加入矩阵
        steps = len(acc_matrix[0])  # step 数

        plt.figure()  # 新图：训练 acc
        for t in range(steps):  # 每个 step 一条线
            y = [acc_matrix[e][t] for e in range(E)]  # 取出该 step 在所有 epoch 的 acc
            plt.plot(epochs, y, label=f"step={t+1}")  # 画线（默认颜色）
        plt.xlabel("epoch")  # x 轴
        plt.ylabel("strict accuracy")  # y 轴（全对才对）
        plt.title("Train Strict Accuracy vs Epoch (one line per step)")  # 标题
        plt.legend()  # 图例
        plt.show()  # 显示

    # ---------- 3) 可选：画 Val Acc vs Epoch（每一步一条线） ----------
    val_acc_seq = history.get("val_acc_per_step", None)  # 取验证 acc 序列（list）
    if val_acc_seq is not None and len(val_acc_seq) > 0:  # 如果存在
        E = len(val_acc_seq)  # epoch 数
        epochs = list(range(1, E + 1))  # epoch 轴

        acc_matrix = []  # 保存每个 epoch 的 acc 向量
        for e in range(E):  # 遍历 epoch
            acc_vec = val_acc_seq[e]  # 当前 epoch 的 val acc_per_step
            acc_vec = acc_vec.detach().cpu().numpy() if hasattr(acc_vec, "detach") else acc_vec  # 转 numpy
            acc_matrix.append(acc_vec)  # 加入矩阵
        steps = len(acc_matrix[0])  # step 数

        plt.figure()  # 新图：验证 acc
        for t in range(steps):  # 每个 step 一条线
            y = [acc_matrix[e][t] for e in range(E)]  # 该 step 在各 epoch 的 acc
            plt.plot(epochs, y, label=f"step={t+1}")  # 画线
        plt.xlabel("epoch")  # x 轴
        plt.ylabel("strict accuracy")  # y 轴
        plt.title("Val Strict Accuracy vs Epoch (one line per step)")  # 标题
        plt.legend()  # 图例
        plt.show()  # 显示


def plot(history: dict):
    # ========= 1) Loss vs Epoch =========
    train_loss = history.get("train_loss", None)
    if train_loss is not None and len(train_loss) > 0:
        epochs = list(range(1, len(train_loss) + 1))

        plt.figure()
        plt.plot(epochs, train_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Train Loss vs Epoch")
        plt.show()

    # ========= helper: list[tensor] -> (E, steps) numpy =========
    def _stack_acc(acc_list):
        if acc_list is None or len(acc_list) == 0:
            return None
        mat = []
        for v in acc_list:
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            mat.append(v)
        return mat  # list of np arrays, each shape=(steps,)

    acc_color_list = history.get("train_acc_per_step_color", None)
    acc_count_list = history.get("train_acc_per_step_count", None)
    acc_color_count_list = history.get("train_acc_per_step_color_count", None)

    acc_color = _stack_acc(acc_color_list)
    acc_count = _stack_acc(acc_count_list)
    acc_color_count = _stack_acc(acc_color_count_list)

    # 如果三个都没有就直接返回
    if acc_color is None and acc_count is None and acc_color_count is None:
        return

    # 统一 epoch 轴长度：以第一个非空为准
    first = acc_color or acc_count or acc_color_count
    E = len(first)
    epochs = list(range(1, E + 1))
    steps = len(first[0])

    # ========= 2) 1×3 子图：三个 head 的 Acc =========
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 行 3 列

    def _plot_one(ax, acc_mat, title):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("strict accuracy")
        if acc_mat is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
        for t in range(steps):
            y = [acc_mat[e][t] for e in range(E)]
            ax.plot(epochs, y, label=f"step={t+1}")
        ax.legend()

    _plot_one(axes[0], acc_color, "Color Presence (Train) - Strict Acc")
    _plot_one(axes[1], acc_count, "Total Count (Train) - Strict Acc")
    _plot_one(axes[2], acc_color_count, "Per-Color Count (Train) - Strict Acc")

    plt.tight_layout()
    plt.show()
