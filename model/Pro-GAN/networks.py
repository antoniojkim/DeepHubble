import torch
import torchx
import numpy as np


class Generator(torchx.nn.Module):
    def __init__(
        self,
        num_channels=3,
        resolution=4,
        fmap_base: int = 2048,
        fmap_decay: float = 1.0,
        fmap_max: int = 256,
        latent_size: int = None,
        use_wscale: bool = True,
        device: torch.device = None,
    ):
        super().__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution >= 4 and resolution == np.exp2(resolution_log2)

        def nf(stage):
            return min(int(fmap_base / np.exp2(stage * fmap_decay)), fmap_max)

        if latent_size is None:
            latent_size = nf(0)

        self.latent_size = latent_size
        self.num_channels = num_channels
        self.resolution = resolution
        self.resolution_log2 = resolution_log2

        self.lod_in = 0

        def upsample2d(factor=2):
            assert isinstance(factor, int) and factor >= 1

            if factor == 1:
                return torch.nn.Identity()

            return torch.nn.Upsample(scale_factor=factor)

        def block(res):
            if res == 2:  # 4x4
                return torch.nn.Sequential(
                    torchx.nn.View(-1, latent_size, 1, 1),
                    torchx.nn.PixelwiseNorm(),
                    torchx.nn.ConvTranspose2d(
                        latent_size,
                        nf(res - 1),
                        4,
                        1,
                        gain=np.sqrt(2) / 4,
                        use_wscale=use_wscale,
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.PixelwiseNorm(),
                    torchx.nn.Conv2d(
                        nf(res - 1), nf(res - 1), 3, 1, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.PixelwiseNorm(),
                )
            else:  # 8x8 and up
                return torch.nn.Sequential(
                    upsample2d(),
                    torchx.nn.Conv2d(
                        nf(res - 2), nf(res - 1), 3, 1, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.PixelwiseNorm(),
                    torchx.nn.Conv2d(
                        nf(res - 1), nf(res - 1), 3, 1, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.PixelwiseNorm(),
                )

        def torgb(res):  # res = 2..resolution_log2
            return torchx.nn.Conv2d(
                nf(res - 1), self.num_channels, 1, 1, gain=1, use_wscale=use_wscale
            )

        self.blocks = torch.nn.ModuleDict(
            {
                f"block_{2**res}x{2**res}": block(res)
                for res in range(2, resolution_log2 + 1)
            }
        )
        self.torgb = torch.nn.ModuleDict(
            {
                f"torgb_{2**res}x{2**res}": torgb(res)
                for res in range(max(resolution_log2 - 1, 2), resolution_log2 + 1)
            }
        )

        self.block_layers = [
            (name, self.blocks[name])
            for name in (
                f"block_{2**res}x{2**res}" for res in range(2, resolution_log2)
            )
        ]
        self.interpolate_layers = [
            self.blocks[f"block_{2**resolution_log2}x{2**resolution_log2}"],
            self.torgb[f"torgb_{2**resolution_log2}x{2**resolution_log2}"],
        ]
        if self.resolution_log2 > 2:
            self.upsample_layers = [
                self.torgb[f"torgb_{2**(resolution_log2-1)}x{2**(resolution_log2-1)}"],
                upsample2d(),
            ]

        self.device = device
        if device is not None:
            self.to(device)

    def forward(self, x, alpha=1):
        if self.resolution_log2 == 2:
            for layer in self.interpolate_layers:
                x = layer(x)
        else:
            for name, block in self.block_layers:
                x = block(x)

            if alpha == 0:
                for layer in self.upsample_layers:
                    x = layer(x)
            elif alpha == 1:
                for layer in self.interpolate_layers:
                    x = layer(x)
            else:
                x1 = x
                for layer in self.upsample_layers:
                    x1 = layer(x1)

                x2 = x
                for layer in self.interpolate_layers:
                    x2 = layer(x2)

                x = (1 - alpha) * x1 + alpha * x2

        return x


class Discriminator(torchx.nn.Module):
    def __init__(
        self,
        resolution=4,
        fmap_base: int = 2048,
        fmap_decay: float = 1.0,
        fmap_max: int = 256,
        latent_size: int = None,
        use_wscale: bool = True,
        device: torch.device = None,
    ):
        super().__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution >= 4 and resolution == np.exp2(resolution_log2)

        self.resolution = resolution
        self.resolution_log2 = resolution_log2

        def nf(stage):
            return min(int(fmap_base / np.exp2(stage * fmap_decay)), fmap_max)

        lod_in = 0

        def downsample2d(factor=2):
            assert isinstance(factor, int) and factor >= 1

            if factor == 1:
                return torch.nn.Identity()

            return torch.nn.AvgPool2d(factor, factor)

        def fromrgb(res):  # res = 2..resolution_log2
            return torch.nn.Sequential(
                torchx.nn.Conv2d(3, nf(res - 1), 1, 1, use_wscale=use_wscale),
                torch.nn.LeakyReLU(0.2, inplace=resolution_log2 > 3),
            )

        def block(res):
            if res >= 3:  # 8x8 and up
                return torch.nn.Sequential(
                    torchx.nn.Conv2d(
                        nf(res - 1), nf(res - 1), 3, 1, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.Conv2d(
                        nf(res - 1), nf(res - 2), 3, 1, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    downsample2d(),
                )
            else:  # 4x4
                return torch.nn.Sequential(
                    torchx.nn.MinibatchStddev(),
                    torchx.nn.Conv2d(
                        nf(res - 1) + 1, nf(res - 1), 1, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.Conv2d(
                        nf(res - 1), nf(res - 2), 4, 1, use_wscale=use_wscale
                    ),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torchx.nn.Conv2d(
                        nf(res - 2), 1, 1, 1, gain=1, use_wscale=use_wscale
                    ),
                )

        self.blocks = torch.nn.ModuleDict(
            {
                f"block_{2**res}x{2**res}": block(res)
                for res in range(2, resolution_log2 + 1)
            }
        )
        self.fromrgb = torch.nn.ModuleDict(
            {
                f"fromrgb_{2**res}x{2**res}": fromrgb(res)
                for res in range(max(resolution_log2 - 1, 2), resolution_log2 + 1)
            }
        )

        self.block_layers = [
            (name, self.blocks[name])
            for name in (
                f"block_{2**res}x{2**res}" for res in range(resolution_log2 - 1, 1, -1)
            )
        ]
        self.interpolate_layers = [
            self.fromrgb[f"fromrgb_{2**resolution_log2}x{2**resolution_log2}"],
            self.blocks[f"block_{2**resolution_log2}x{2**resolution_log2}"],
        ]
        if self.resolution_log2 > 2:
            self.downsample_layers = [
                downsample2d(),
                self.fromrgb[
                    f"fromrgb_{2**(resolution_log2-1)}x{2**(resolution_log2-1)}"
                ],
            ]

        self.device = device
        if device is not None:
            self.to(device)

    def forward(self, x, alpha=1):
        if self.resolution_log2 == 2:
            for layer in self.interpolate_layers:
                x = layer(x)
        else:
            if alpha == 0:
                for layer in self.downsample_layers:
                    x = layer(x)
            elif alpha == 1:
                for layer in self.interpolate_layers:
                    x = layer(x)
            else:
                x1 = x
                for layer in self.downsample_layers:
                    x1 = layer(x1)

                x2 = x
                for layer in self.interpolate_layers:
                    x2 = layer(x2)

                x = torchx.utils.lerp(x1, x2, alpha)

            for name, block in self.block_layers:
                x = block(x)

        return torch.flatten(x, 0)
