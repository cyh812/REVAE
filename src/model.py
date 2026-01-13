import torch
import torch.nn as nn
import torch.nn.functional as F


# 第一版
class REVAE_V1(nn.Module):
    def __init__(
        self,
        z_color_dim=16,
        z_shape_dim=16,
        z_count_dim=16,
        n_colors=8,
        n_shapes=3,
        n_count_classes=11, #数量从0~10，但数量为0其实不存在的
    ):
        super().__init__()

        # 这四个维度一样
        self.z_color_dim = z_color_dim
        self.z_shape_dim = z_shape_dim
        self.z_count_dim = z_count_dim
        self.z_total_dim = z_color_dim

        # -----------------------
        # Encoder: 224 -> 7
        # -----------------------
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1),  # /2
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        
        def deconv_block(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),  # x2
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            conv_block(3, 32),     # B,32,112,112
            conv_block(32, 64),    # B,64,56,56
            conv_block(64, 128),   # B,128,28,28
            conv_block(128, 256),  # B,256,14,14
            conv_block(256, 512),  # B,512,7,7
        )

        self.enc_fc = nn.Sequential(
            nn.Flatten(),                          # (B, 512*7*7)
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )

        # posterior heads (mu/logvar) for each group
        self.mu_color = nn.Linear(512, z_color_dim)
        self.lv_color = nn.Linear(512, z_color_dim)

        self.mu_shape = nn.Linear(512, z_shape_dim)
        self.lv_shape = nn.Linear(512, z_shape_dim)

        self.mu_count = nn.Linear(512, z_count_dim)
        self.lv_count = nn.Linear(512, z_count_dim)

        # -----------------------
        # Decoder: 7 -> 224
        # -----------------------
        self.dec_fc = nn.Sequential(
            nn.Linear(self.z_total_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            deconv_block(512, 256),  # 7 -> 14
            deconv_block(256, 128),  # 14 -> 28
            deconv_block(128, 64),   # 28 -> 56
            deconv_block(64, 32),    # 56 -> 112
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 112 -> 224
        )
        # 输出用 logits；重建时可用 BCEWithLogitsLoss 或先 sigmoid 后 MSE

        # -----------------------
        # Heads (supervision)
        # -----------------------
        self.color_head = nn.Linear(z_color_dim, n_colors)           # multi-label logits
        self.shape_head = nn.Linear(z_shape_dim, n_shapes)           # multi-label logits
        self.count_head = nn.Linear(z_count_dim, n_count_classes)    # class logits

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 放在类里面方便组织的普通函数
    @staticmethod
    def kl_standard_normal(mu, logvar):
        # KL( N(mu,var) || N(0,I) )
        # returns (B,)
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.enc_fc(h)

        mu_c, lv_c = self.mu_color(h), self.lv_color(h)
        mu_s, lv_s = self.mu_shape(h), self.lv_shape(h)
        mu_n, lv_n = self.mu_count(h), self.lv_count(h)

        z_c = self.sample(mu_c, lv_c)
        z_s = self.sample(mu_s, lv_s)
        z_n = self.sample(mu_n, lv_n)

        return (mu_c, lv_c, z_c), (mu_s, lv_s, z_s), (mu_n, lv_n, z_n)

    def decode(self, z_c, z_s, z_n):
        z = z_c + z_s + z_n
        h = self.dec_fc(z).view(-1, 512, 7, 7)
        x_logits = self.decoder(h)  # (B,3,224,224)
        return x_logits

    def forward(self, x):
        (mu_c, lv_c, z_c), (mu_s, lv_s, z_s), (mu_n, lv_n, z_n) = self.encode(x)
        x_logits = self.decode(z_c, z_s, z_n)

        heads = {
            "color_logits": self.color_head(z_c),   # (B,8)
            "shape_logits": self.shape_head(z_s),   # (B,3)
            "count_logits": self.count_head(z_n),   # (B,11)
        }
        post = {
            "mu_color": mu_c, "lv_color": lv_c, "z_color": z_c,
            "mu_shape": mu_s, "lv_shape": lv_s, "z_shape": z_s,
            "mu_count": mu_n, "lv_count": lv_n, "z_count": z_n,
        }
        return x_logits, post, heads
