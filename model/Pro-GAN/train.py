import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchx
from tqdm import tqdm

from dataset import Dataset
from networks import Generator, Discriminator


def generate_noise(generator, minibatch_size=1):
    return torch.tanh(
        torch.randn(minibatch_size, generator.latent_size, device=generator.device)
    )


def check_generator_dims(generator, res):
    noise = generate_noise(generator)
    for alpha in (0, 0.5, 1):
        fake_output = generator.forward(noise, alpha)

        err = (
            f"generator shape does not match: {fake_output.shape} != {(1, 3, res, res)}"
        )
        assert fake_output.shape == (1, 3, res, res), err


def check_discriminator_dims(discriminator, res):
    with torch.autograd.set_detect_anomaly(True):
        imgs = torch.sigmoid(torch.rand([4, 3, res, res], device=discriminator.device))
        zeros = torch.zeros(4, device=discriminator.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        for alpha in (0, 0.5, 1):
            prediction = discriminator.forward(imgs, alpha)
            assert prediction.shape == (4,), f"{prediction.shape} != (4,)"

            err = criterion(prediction, zeros)
            err.backward()
            assert any(
                param.grad.sum().item() != 0
                for name, param in discriminator.blocks.named_parameters()
                if param.grad is not None
            )
            discriminator.zero_grad()


def FromTensor(tensor):
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1

    return np.moveaxis(tensor.detach().cpu().numpy(), 1, -1)


def generate_fake_images(generator, params, suffix=""):
    res = params.resolution

    noise = generate_noise(generator, 4)
    fake_output = generator.forward(noise, alpha=0)
    fake_output = FromTensor(fake_output)

    fig, ax = plt.subplots(2, 2, figsize=(10, 11))
    fig.tight_layout()
    ax[0][0].imshow(fake_output[0])
    ax[0][1].imshow(fake_output[1])
    ax[1][0].imshow(fake_output[2])
    ax[1][1].imshow(fake_output[3])
    fig.suptitle(f"Generator ${res} \\times {res}$ {suffix.title()}")
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            params.log_path,
            f"pro-gan_generator_{res}x{res}_{'_'.join(suffix.lower().split())}.png",
        )
    )
    plt.close()


def train(args):
    params = torchx.params.Parameters(args.param_file)

    resolution = params.resolution
    print(f"Training {resolution}x{resolution} DeepHubble Pro-GAN")

    device = torch.device(f"cuda:{params.cuda}" if params.use_gpu else "cpu")
    print("Training device: ", device)

    print(f"\nGenerator {resolution}x{resolution}:")
    generator = Generator(resolution=resolution, device=device)
    print("    Number of Parameters: ", generator.num_params())
    check_generator_dims(generator, res=resolution)

    model_file = os.path.join(
        params.save_model_path,
        f"final_pro-gan_generator_{resolution//2}x{resolution//2}.pt",
    )
    if os.path.isfile(model_file):
        generator.load(model_file)
        print("    Loaded Generator: ", model_file)

    print(f"\nDiscriminator {resolution}x{resolution}:")
    discriminator = Discriminator(resolution=resolution, device=device)
    print("    Number of Parameters: ", discriminator.num_params())
    check_discriminator_dims(discriminator, res=resolution)

    model_file = os.path.join(
        params.save_model_path,
        f"final_pro-gan_discriminator_{resolution//2}x{resolution//2}.pt",
    )
    if os.path.isfile(model_file):
        discriminator.load(model_file)
        print("    Loaded Discriminator: ", model_file)

    print("\n"+"-"*80+"\n")

    generate_fake_images(generator, params, "Untrained")

    dataloader = torch.utils.data.DataLoader(
        Dataset(resolution=resolution, size=800000),
        batch_size=params.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )

    generator_losses = []
    discriminator_losses = []

    iteration = 0

    G_criterion = torchx.nn.WGAN_ACGAN()
    D_criterion = torchx.nn.WGANGP_ACGAN(generator, discriminator, use_gp=True)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=params.learning_rate, betas=(0, 0.99)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=params.learning_rate, betas=(0, 0.99)
    )

    def train(fade_in: bool):
        nonlocal iteration

        generator.train()
        discriminator.train()

        start = time.time()

        with tqdm(total=len(dataloader), leave=True) as progress:

            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)

                alpha = i / len(dataloader) if fade_in else 1

                noise = generate_noise(generator, imgs.shape[0])
                fake_images = generator.forward(noise, alpha)

                discriminator_err = D_criterion(imgs, fake_images.detach(), alpha)
                discriminator.zero_grad()
                discriminator_err.backward()
                discriminator_optimizer.step()

                generator_err = G_criterion(
                    discriminator.forward(fake_images, alpha), None
                )
                generator.zero_grad()
                generator_err.backward()
                generator_optimizer.step()

                progress.set_postfix(
                    dloss="%.6f" % discriminator_err.item(),
                    gloss="%.6f" % generator_err.item(),
                    alpha="%.3f" % alpha,
                )
                progress.update()

                if abs(discriminator_err.item()) > 50:
                    raise Exception("Training has diverged.")

                generator_losses.append(generator_err.item())
                discriminator_losses.append(discriminator_err.item())

                iteration += 1

                if iteration % params.save_interval == 0:
                    time.sleep(2)

                    generator.save(
                        os.path.join(
                            params.save_model_path,
                            f"latest_pro-gan_generator_{resolution}x{resolution}.pt",
                        )
                    )
                    discriminator.save(
                        os.path.join(
                            params.save_model_path,
                            f"latest_pro-gan_discriminator_{resolution}x{resolution}.pt",
                        )
                    )

                    generate_fake_images(generator, params, f"Pro-GAN Iteration {iteration}")

                    pd.DataFrame(
                        {
                            "Generator": generator_losses,
                            "Discriminator": discriminator_losses,
                        }
                    ).rolling(10).mean().plot(title="Training Losses (Smoothing: 10)")
                    plt.savefig(
                        os.path.join(
                            params.log_path,
                            f"generator_{resolution}x{resolution}_training_losses_s10.png",
                        )
                    )
                    plt.close()

                    pd.DataFrame(
                        {
                            "Generator": generator_losses,
                            "Discriminator": discriminator_losses,
                        }
                    ).rolling(100).mean().plot(title="Training Losses (Smoothing: 100)")
                    plt.savefig(
                        os.path.join(
                            params.log_path,
                            f"generator_{resolution}x{resolution}_training_losses_s100.png",
                        )
                    )
                    plt.close()

                    time.sleep(2)

        end = time.time()

        with open(os.path.join(params.log_path, "time.log"), "a") as file:
            file.write(f"Training time {resolution}x{resolution} (fade_in={fade_in}): {end - start}\n")

    if resolution > 4:
        train(fade_in=True)

    train(fade_in=False)

    generate_fake_images(generator, params, "Trained")

    generator.save(
        os.path.join(
            params.save_model_path,
            f"final_pro-gan_generator_{resolution}x{resolution}.pt",
        )
    )
    discriminator.save(
        os.path.join(
            params.save_model_path,
            f"final_pro-gan_discriminator_{resolution}x{resolution}.pt",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-file", default="params.yml")
    train(parser.parse_args())
