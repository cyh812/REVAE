import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MLR(nn.Module):
    def __init__(
        self,
        z_color=128,
        z_shape=128,
        n_colors=8,
        n_shapes=3,
    ):
        super().__init__()

        self.z_color = z_color
        self.z_shape = z_shape

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

        def mlp_block(cin, cout):
            return nn.Sequential(
                nn.Linear(cin, cout),
                nn.ReLU(inplace=True)
            )

        # __init__ 里：把 encoder/decoder 拆开
        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)   # 这里输出 14x14，做 skip
        self.skip = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc5 = conv_block(256, 512)   # 输出 7x7

        self.dec1 = deconv_block(512, 256) # 7->14

        self.dec2 = deconv_block(256, 128) # 14->28
        self.dec3 = deconv_block(128, 64)  # 28->56
        self.dec4 = deconv_block(64, 32)   # 56->112
        self.dec5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) # 112->224


        self.enc_fc = nn.Sequential(
            nn.Flatten(),                          # (B, 512*7*7)
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )

        self.mu_color = nn.Linear(512, z_color)
        self.lv_color = nn.Linear(512, z_color)
        self.mu_shape = nn.Linear(512, z_shape)
        self.lv_shape = nn.Linear(512, z_shape)


        self.dec_fc = nn.Sequential(
            nn.Linear(z_color, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512*7*7),
            nn.ReLU(inplace=True),
        )

        self.color_head = nn.Linear(z_color, n_colors)           # multi-label logits
        self.shape_head = nn.Linear(z_shape, n_shapes)           # multi-label logits

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
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)          # (B,256,14,14) 作为 skip
        skip = self.skip(h4)
        h5 = self.enc5(h4)          # (B,512,7,7)

        h = self.enc_fc(h5)         # (B,512)
        mu_color, lv_color = self.mu_color(h), self.lv_color(h)
        mu_shape, lv_shape = self.mu_shape(h), self.lv_shape(h)
        return mu_color, lv_color, mu_shape, lv_shape, skip    # 多返回一个 skip
    

    def decode(self, z_color, z_shape, skip, drop_path):
        z = z_color + z_shape
        B = z.size(0)
        h = self.dec_fc(z).view(B, 512, 7, 7)
        h = self.dec1(h)            # (B,256,14,14)

        if drop_path == "skip-only":
            h = skip
        if drop_path == "latent-only":
            h = h
        if drop_path == "latent+skip":
            h = h + skip

        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        x_logits = self.dec5(h)

        if drop_path != "skip-only":
            heads = {
                "color_logits": self.color_head(z_color),   # (B,8)
                "shape_logits": self.shape_head(z_shape),   # (B,3)
            }
        else:
            heads = {}
        return x_logits, heads

    def forward(self, x, drop_path):
        # drop_path = "latent-only"/"skip-only"/"latent+skip"
        mu_color, lv_color, mu_shape, lv_shape, skip = self.encode(x)
        z_color = self.sample(mu_color, lv_color)
        z_shape = self.sample(mu_shape, lv_shape)

        x_logits, heads = self.decode(z_color, z_shape, skip, drop_path)
        post = {
            "mu_color": mu_color, "lv_color": lv_color, "z_color": z_color,
            "mu_shape": mu_shape, "lv_shape": lv_shape, "z_shape": z_shape,
        }
        return x_logits, heads, post

    def predict(self, x, pre_type):
        """
        pre_type
        """
        mu_color, lv_color, mu_shape, lv_shape, skip = self.encode(x)
        z_color = self.sample(mu_color, lv_color)
        z_shape = self.sample(mu_shape, lv_shape)

        if pre_type == "color-only":
            z = z_color
            B = z.size(0)
            h = self.dec_fc(z).view(B, 512, 7, 7)
            h = self.dec1(h)
        if pre_type == "shape-only":
            z = z_shape
            B = z.size(0)
            h = self.dec_fc(z).view(B, 512, 7, 7)
            h = self.dec1(h)
        if pre_type == "skip-only":
            h = skip
        if pre_type == "latent-only":
            z = z_color + z_shape
            B = z.size(0)
            h = self.dec_fc(z).view(B, 512, 7, 7)
            h = self.dec1(h)            # (B,256,14,14)
        if pre_type == "latent+skip":
            z = z_color + z_shape
            B = z.size(0)
            h = self.dec_fc(z).view(B, 512, 7, 7)
            h = self.dec1(h)            # (B,256,14,14)
            h = h + skip

        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        x_logits = self.dec5(h)

        if pre_type != "skip-only":
            heads = {
                "color_logits": self.color_head(z_color),   # (B,8)
                "shape_logits": self.shape_head(z_shape),   # (B,3)
            }
        else:
            heads = {}
        return x_logits, heads