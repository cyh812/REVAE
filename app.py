import gradio as gr
import numpy as np
import os
import matplotlib.pyplot as plt

# 让本 Notebook 进程里的 HTTP 客户端不走代理（含 httpx/requests）
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

# 关键：保证 localhost/127.0.0.1 永远直连
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

COLOR_NAMES = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink"]

def dummy_predict(_img):
    classes = ["Tiger", "Puma", "Leopard", "Lion", "Lynx"]
    probs = np.random.dirichlet(np.ones(len(classes))).tolist()
    conf = {c: float(p) for c, p in sorted(zip(classes, probs), key=lambda x: -x[1])}

    presence = np.random.randint(0, 2, size=8)
    counts = np.random.randint(0, 20, size=8) * presence

    fig1 = plt.figure()
    plt.bar(COLOR_NAMES, presence)
    plt.ylim(-0.1, 1.1)
    plt.title("Color Presence (0/1)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fig2 = plt.figure()
    plt.bar(COLOR_NAMES, counts)
    plt.title("Per-color Counts")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    return conf, fig1, fig2


def build_ui():
    with gr.Blocks(title="UI Prototype") as demo:
        gr.Markdown("# 重入模型展示")

        # ========= Top Control Panel =========
        with gr.Row(equal_height=True):
            # ---- Left: Model/Dataset/Sampling ----
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("## Selection")

                with gr.Row():
                    model_dd = gr.Dropdown(
                        label="Model",
                        choices=["Model-A", "Model-B", "Model-C"],
                        value="Model-A",
                        interactive=True,
                    )

                    dataset_dd = gr.Dropdown(
                        label="Dataset",
                        choices=["CLEVR-train", "CLEVR-val", "Custom-set"],
                        value="CLEVR-val",
                        interactive=True,
                    )


                with gr.Row():
                    sampling_mode = gr.Radio(
                    label="Sampling Mode",
                    choices=["Single", "Batch"],
                    value="Single",
                    interactive=True,
                    )

                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1, maximum=64, step=1, value=8,
                        interactive=True,
                    )
                    image_index = gr.Slider(
                        label="Image Index",
                        minimum=0, maximum=9999, step=1, value=0,
                        interactive=True,
                    )

                with gr.Row():
                    mask_mode = gr.Radio(
                        label="Mask Mode (mutually exclusive)",
                        choices=["Mode-1", "Mode-2", "Mode-3"],
                        value="Mode-1",
                        interactive=True,
                    )

                    # Only ONE slider now; its meaning depends on mask_mode
                    mask_strength = gr.Slider(
                        label="Mask Strength / Threshold",
                        minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                        interactive=True,
                    )
                
                with gr.Row():
                    btn_load = gr.Button("Load Sample", variant="secondary")
                    btn_run = gr.Button("Run Inference", variant="primary")

            # ---- Right: Mask Mode (mutually exclusive) ----
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("## Images")

                with gr.Row():
                    img_input = gr.Image(label="Input Image", type="numpy")
                    img_output = gr.Image(label="Output / Overlay / Mask", type="numpy")
                
                with gr.Row():
                    image_index = gr.Slider(
                        label="Image Index",
                        minimum=0, maximum=9999, step=1, value=0,
                        interactive=True,
                    )
                    btn_run = gr.Button("Run Inference", variant="primary")

        gr.Markdown("---")

        # ========= Visualization: 3 plots in a row =========
        gr.Markdown("## Prediction Visualization")
        with gr.Row():
            out_label = gr.Label(num_top_classes=5, label="Prediction")
            plot_color_pre = gr.Plot(label="Color Presence (8 colors, binary)")
            plot_color_cous = gr.Plot(label="Per-color Counts (Bar Chart)")
            plot_color_presence = gr.Plot(label="Color Presence (8 colors, binary)")
            plot_color_counts = gr.Plot(label="Per-color Counts (Bar Chart)")

        btn_run.click(
            fn=dummy_predict,
            inputs=[img_input],   # img_input 可以为空也没问题
            outputs=[out_label, plot_color_presence, plot_color_counts],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)

