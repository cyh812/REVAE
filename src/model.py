import torch
import torch.nn as nn
import torch.nn.functional as F


# 第一版
class REVAE_V1(nn.Module):
    def __init__(
        self,
        z_color_dim=32,
        z_shape_dim=32,
        z_count_dim=32,
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
                # nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        
        def deconv_block(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),  # x2
                # nn.BatchNorm2d(cout),
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


class TDVAE(nn.Module):
    """
    A PyTorch implementation matching the architecture we discussed:

    x [B,64,64,3]
      -> 3*Conv (stride=2) => feat [B,128,8,8]
      -> Flatten + MLP => hiddens [B,256]
      -> q(y|x): 2*(FC+ReLU) + FC(linear) => (mu_y, logvar_y) with y_dim=10
      -> q(z|x,y): y->emb(64) ; concat([hiddens, y_emb]) => [B,320]
                 -> 2*(FC+ReLU) + FC(linear) => (mu_z, logvar_z) with z_dim=32
      -> p(z|y): 2*(FC+ReLU) + FC(linear) => (mu_z_prior, logvar_z_prior)
      -> p(x|z): 2*(FC+ReLU) -> reshape [B,128,8,8] -> upsample+conv blocks -> x_hat [B,3,64,64]

    Notes:
    - Uses diagonal Gaussians (Normal) parameterized by (mu, logvar).
    - Returns a dict with all key tensors so you can build losses externally.
    """

    def __init__(
        self,
        y_dim: int = 10,
        z_dim: int = 32,
        hidden_dim: int = 256,
        y_emb_dim: int = 64,
        img_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.y_emb_dim = y_emb_dim
        self.img_channels = img_channels

        # -------- Shared conv encoder: [B,3,64,64] -> [B,128,8,8] --------
        # Conv1: 64 -> 32
        # Conv2: 32 -> 16
        # Conv3: 16 -> 8
        self.enc_conv = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_feat_channels = base_channels * 4  # 128 if base_channels=32
        self.enc_feat_hw = 8
        self.enc_feat_dim = self.enc_feat_channels * self.enc_feat_hw * self.enc_feat_hw  # 128*8*8=8192

        # Flatten + MLP -> hiddens [B,256]
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.enc_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # -------- q(y|x): hiddens -> (mu_y, logvar_y) --------
        self.qy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * y_dim),  # outputs [mu_y, logvar_y]
        )

        # -------- q(z|x,y): y -> y_emb (64), concat with hiddens -> (mu_z, logvar_z) --------
        self.y_to_emb = nn.Sequential(
            nn.Linear(y_dim, y_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.qz = nn.Sequential(
            nn.Linear(hidden_dim + y_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * z_dim),  # outputs [mu_z, logvar_z]
        )

        # -------- p(z|y): y -> (mu_z_prior, logvar_z_prior) --------
        self.pz = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

        # -------- p(x|z): z -> [B,128,8,8] -> upsample+conv to [B,3,64,64] --------
        self.dec_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.enc_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Upsample + Conv blocks (resize+conv)
        # 8 -> 16 -> 32 -> 64
        self.dec_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.enc_feat_channels, self.enc_feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.enc_feat_channels, self.enc_feat_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.enc_feat_channels // 2, self.enc_feat_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec_out = nn.Conv2d(self.enc_feat_channels // 4, img_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + std * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode_shared(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,64,64]
        returns hiddens: [B,256]
        """
        feat = self.enc_conv(x)  # [B,128,8,8]
        feat_flat = feat.flatten(1)  # [B,8192]
        h = self.enc_mlp(feat_flat)  # [B,256]
        return h

    def infer_y(self, hiddens: torch.Tensor):
        """
        hiddens: [B,256]
        returns mu_y, logvar_y, y_sample
        """
        logits = self.qy(hiddens)  # [B, 2*y_dim]
        mu_y, logvar_y = torch.chunk(logits, 2, dim=1)
        y = self.reparameterize(mu_y, logvar_y)
        return mu_y, logvar_y, y

    def infer_z_posterior(self, hiddens: torch.Tensor, y: torch.Tensor):
        """
        q(z|x,y)
        hiddens: [B,256]
        y: [B,10]
        returns mu_z, logvar_z, z_post
        """
        y_emb = self.y_to_emb(y)  # [B,64]
        hy = torch.cat([hiddens, y_emb], dim=1)  # [B,320]
        logits = self.qz(hy)  # [B, 2*z_dim]
        mu_z, logvar_z = torch.chunk(logits, 2, dim=1)
        z = self.reparameterize(mu_z, logvar_z)
        return mu_z, logvar_z, z

    def infer_z_prior(self, y: torch.Tensor):
        """
        p(z|y)
        y: [B,10]
        returns mu_z_prior, logvar_z_prior, z_prior_sample
        """
        logits = self.pz(y)  # [B, 2*z_dim]
        mu, logvar = torch.chunk(logits, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode_x(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B,32]
        returns x_logits: [B,3,64,64]
        (Apply sigmoid outside if you want Bernoulli mean)
        """
        flat = self.dec_mlp(z)  # [B,8192]
        feat = flat.view(-1, self.enc_feat_channels, self.enc_feat_hw, self.enc_feat_hw)  # [B,128,8,8]
        feat = self.dec_up1(feat)  # [B,128,16,16]
        feat = self.dec_up2(feat)  # [B,64,32,32]
        feat = self.dec_up3(feat)  # [B,32,64,64]
        x_logits = self.dec_out(feat)  # [B,3,64,64]
        return x_logits

    def forward(self, x: torch.Tensor, sample_from: str = "posterior"):
        """
        x: [B,3,64,64]
        sample_from: "posterior" or "prior"
        Returns a dict containing:
          hiddens, (mu_y, logvar_y, y),
          (mu_z_post, logvar_z_post, z_post),
          (mu_z_prior, logvar_z_prior, z_prior),
          x_logits (decoded from chosen z)
        """
        h = self.encode_shared(x)

        mu_y, logvar_y, y = self.infer_y(h)
        mu_z_post, logvar_z_post, z_post = self.infer_z_posterior(h, y)
        mu_z_pr, logvar_z_pr, z_pr = self.infer_z_prior(y)

        z_used = z_post if sample_from.lower() == "posterior" else z_pr
        x_logits = self.decode_x(z_used)

        return {
            "hiddens": h,
            "mu_y": mu_y,
            "logvar_y": logvar_y,
            "y": y,
            "mu_z_post": mu_z_post,
            "logvar_z_post": logvar_z_post,
            "z_post": z_post,
            "mu_z_prior": mu_z_pr,
            "logvar_z_prior": logvar_z_pr,
            "z_prior": z_pr,
            "z_used": z_used,
            "x_logits": x_logits,
        }
