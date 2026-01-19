import torch
import matplotlib.pyplot as plt



# 绘制loss曲线


# 统计一批图：原图+完全重建+色形重建+纯色重建+纯形重建+skip重建
def show_img(t):
    # t: (3,H,W) in [0,1]
    t = t.permute(1, 2, 0).numpy()
    plt.imshow(t)
    plt.axis("off")

def generate_reconstructions(model, x, device):
    # 定义不同的 drop_path 模式
    drop_paths = ["latent+skip", "latent-only", "color-only", "shape-only", "skip-only"]
    reconstructions = []

    for drop_path in drop_paths:
        with torch.no_grad():
            x_logits, heads = model.predict(x, drop_path=drop_path)
            recon = torch.sigmoid(x_logits)[0].detach().cpu()  # (3, H, W)
            reconstructions.append(recon)

    return reconstructions

def display_reconstructed_images(dataloader, model, num_images, device):
    # 从数据集中随机选择指定数量的图片
    images, color_mh, shape_mh, count_oh, img_fn = next(iter(dataloader))
    images = images.to(device)

    fig, axes = plt.subplots(num_images, 6, figsize=(12, num_images * 2))

    for i in range(num_images):
        # 获取每张图像
        x = images[i].unsqueeze(0)

        # 获取不同的重建版本
        reconstructions = generate_reconstructions(model, x, device)

        # 原图
        axes[i, 0].set_title("Input")
        show_img(images[i].detach().cpu())  # 显示原图
        axes[i, 0].imshow(images[i].detach().cpu().permute(1, 2, 0).numpy())
        axes[i, 0].axis("off")

        # 重建图
        for j, recon in enumerate(reconstructions):
            axes[i, j + 1].set_title(drop_paths[j])  # 给每个重建加标题
            axes[i, j + 1].imshow(recon.permute(1, 2, 0))  # 显示重建图
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.show()

# 示例：调用 display_reconstructed_images 函数来显示重建图
# 假设你已经定义了 dataloader, model, device，并且可以获取 `train_loader`
num_images = 3  # 设置想要显示的图片数量
display_reconstructed_images(train_loader, revae, num_images, device)


# 统计一批图：数量+形状分类