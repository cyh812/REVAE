import torch
import torch.nn as nn
import torch.nn.functional as F

class REVAE(nn.Module):
    def __init__(
        self,
        z_latent=128,
        skip_mode="add"
    ):
        super().__init__()

        self.z_latent = z_latent
        self.skip_mode = skip_mode

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

        if self.skip_mode == "concat":
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(256 + 256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.skip_fuse = None

        self.dec2 = deconv_block(256, 128) # 14->28
        self.dec3 = deconv_block(128, 64)  # 28->56
        self.dec4 = deconv_block(64, 32)   # 56->112
        self.dec5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) # 112->224


        self.enc_fc = nn.Sequential(
            nn.Flatten(),                          # (B, 512*7*7)
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
        )

        self.mu_latent = nn.Linear(512, z_latent)
        self.lv_latent = nn.Linear(512, z_latent)


        self.dec_fc = nn.Sequential(
            nn.Linear(z_latent, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512*7*7),
            nn.ReLU(inplace=True),
        )

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

    def _apply_skip(self, h14, skip14):
        """h14: (B,256,14,14), skip14: (B,256,14,14)"""
        if self.skip_mode == "none":
            return h14
        if self.skip_mode == "add":
            return h14 + skip14
        # concat
        return self.skip_fuse(torch.cat([h14, skip14], dim=1))

    def encode(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)          # (B,256,14,14) 作为 skip
        skip = self.skip(h4)
        h5 = self.enc5(h4)          # (B,512,7,7)

        h = self.enc_fc(h5)         # (B,512)
        mu, lv = self.mu_latent(h), self.lv_latent(h)
        z = self.sample(mu, lv)
        return mu, lv, z, skip    # 多返回一个 skip
    
    def decode(self, z, skip14):
        B = z.size(0)
        h = self.dec_fc(z).view(B, 512, 7, 7)

        h = self.dec1(h)            # (B,256,14,14)
        if skip14 is not None:
            h = self._apply_skip(h, skip14)

        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        x_logits = self.dec5(h)
        return x_logits

    def forward(self, x):
        mu, logvar, z, skip14 = self.encode(x)
        x_logits = self.decode(z, skip14)
        post = {"mu": mu, "lv": logvar, "z_latent": z}
        return x_logits, post

class REVAE_V2(nn.Module):
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