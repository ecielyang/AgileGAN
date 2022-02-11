import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from stylegan2.stylegan2_pytorch.model import Generator, MLP
from stylegan2.stylegan2_pytorch.model import Discriminator
from torch.utils.data import DataLoader
from criteria.lpips.lpips import LPIPS
from torchvision import utils, transforms
from stylegan2.stylegan2_pytorch.train import g_path_regularize, make_noise, mixing_noise, d_r1_loss
from configs import transforms_config
from datasets.images_dataset import ImagesDataset
from stylegan2.stylegan2_pytorch.non_leaking import augment, AdaptiveAugment

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(log_dir='outcome/log8')


def adv_loss(y_fake, y_real):
    # adv_loss = torch.min(0, -1 + y_real) + torch.min(0, -1 - y_fake)
    # return adv_loss
    real_loss = F.softplus(-y_real)
    fake_loss = F.softplus(y_fake)

    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def configure_datasets():
    transforms_dict = transforms_config.EncodeTransforms.get_transforms()
    train_dataset = ImagesDataset(source_root='/home/ai/project/dataset/agile_train',
                                  source_transform=transforms_dict['transform_gt_train']
                                  )
    test_dataset = ImagesDataset(source_root='/home/ai/project/dataset/portrait2_test',
                                 source_transform=transforms_dict['transform_gt_train']
                                 )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    return train_dataset, test_dataset


def train():
    G_path = "/home/ai/project/agile3/pretrained_models/stylegan2-ffhq-config-f.pt"
    device0 = 'cuda:0'
    device1 = 'cuda:1'
    batch_size = 2
    output_size = 1024
    learning_rate = 0.0002
    w1 = 1
    w2 = 0.4

    # load generator0 and genetator t
    G0 = Generator(output_size, 512, 8).to(device0)
    Gt = Generator(output_size, 512, 8).to(device1)
    checkpoint = torch.load(G_path)
    G0.load_state_dict(checkpoint["g_ema"], strict=False)
    Gt.load_state_dict(checkpoint["g_ema"], strict=False)
    mlp = MLP(1024, 512, 8).to(device0)
    mlp.load_state_dict(checkpoint["g_ema"], strict=False)
    # load discriminator
    D = Discriminator(size=1024).to(device1)
    D.load_state_dict(checkpoint["d"], strict=True)

    train_dataset, test_dataset = configure_datasets()
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True,
                              )

    lpips = LPIPS(net_type='vgg').to(device0).eval()  # DISCARD 9 LAYERS

    loss_dict = {}
    num_epochs = 1000
    Gt_optim = optim.Adam(Gt.parameters(), lr=learning_rate)
    D_optim = optim.Adam(D.parameters(), lr=learning_rate)

    number = 0
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            number += 1
            real_img = batch

            real_img = real_img.to(device0)
            real_img = F.interpolate(real_img, size=[1024, 1024])

            # Update discriminator
            requires_grad(Gt, False)
            requires_grad(G0, False)
            requires_grad(D, True)

            # noise = mixing_noise(batch_size, latent_dim=512, prob=0.9, device=device0)
            noise = torch.randn(batch_size, 18, 512).to(device0).unbind(0)
            z = mlp(noise)
            z = torch.stack(z, dim=0).to(device1)
            fake_img_t = Gt(z, input_is_latent=True, noise=None, return_latents=False)
            fake_img_t = fake_img_t.detach()

            #fake_img_t256 = F.interpolate(fake_img_t, size = [256, 256])
            y_fake = D(fake_img_t)
            y_real = D(real_img.to(device1))

            loss_adv = adv_loss(y_fake, y_real)
            D_loss = loss_adv
            loss_dict["d"] = D_loss.to(device0)

            D.zero_grad()
            D_loss.backward()
            D_optim.step()


            real_img.requires_grad = True
            y_real = D(real_img.to(device1))
            r1_loss = d_r1_loss(y_real, real_img)
            D.zero_grad()
            r1_loss.backward()
            D_optim.step()
            loss_dict["r1"] = r1_loss.to(device0)



            # Update generator
            requires_grad(Gt, True)
            requires_grad(G0, False)
            requires_grad(D, False)


            z0 = torch.randn(batch_size, 18, 512, device=device1)
            z = mlp(z0.to(device0))
            z = torch.stack(z, dim=0)
            fake_img_t = Gt(z.to(device1), input_is_latent=True, noise=None, return_latents=False)
            #fake_img_t256 = F.interpolate(fake_img_t, size=[256, 256])

            y_fake = D(fake_img_t)
            nonsaturating_loss = g_nonsaturating_loss(y_fake).to(device0)
            loss_dict["nonsa_loss"] = nonsaturating_loss

            fake_img_0 = G0(z, input_is_latent=True, noise=None).detach()
            lpips_loss = lpips(fake_img_t.to(device0), fake_img_0).to(device0)
            loss_dict["lpip_loss"] = lpips_loss

            if number % 20 == 0:
                image = torch.cat((fake_img_0.to(device0), fake_img_t.to(device0)), 0)
                utils.save_image(
                    image,
                    "outcome/images8/" + str(number) + "output.jpg",
                    nrow=2,
                    normalize=True,
                    range=(-1, 1),
                )


            Gt_loss =  w1 * lpips_loss + w2 * nonsaturating_loss
            loss_dict["g"] =  Gt_loss

            Gt.zero_grad()
            Gt_loss.backward()
            Gt_optim.step()


            z0 = torch.randn(batch_size, 18, 512, device=device0)
            z = mlp(z0)
            z = torch.stack(z, dim=0).to(device1)
            fake_img_t, latents = Gt(z, input_is_latent=True, noise=None, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img_t, latents, mean_path_length=0)

            Gt.zero_grad()
            path_loss = 2 * path_loss
            loss_dict["path"] = path_loss.to(device0)
            path_loss.backward()
            Gt_optim.step()



            if number % 20 == 0:
                print("step:   ", number)
                print("d loss:      ", loss_dict["d"].item())
                print("g loss:     ", loss_dict["g"].item())
                print("nonsa loss:   ", loss_dict["nonsa_loss"].item())
                print("lpip loss:   ", loss_dict["lpip_loss"].item())
                print("r1 loss:   ", loss_dict["r1"].item())
                print("path loss:   ", loss_dict["path"].item())

                values = loss_dict.values()
                print("loss:        ", loss_dict["g"].item() + loss_dict["d"].item())
                print()

                writer.add_scalar('Loss/train/total', loss_dict["g"].item() + loss_dict["d"].item(), number)
                writer.add_scalar('Loss/train/total', loss_dict["nonsa_loss"].item(), number)
                writer.add_scalar('Loss/train/total', loss_dict["lpip_loss"].item(), number)
                writer.add_scalar('Loss/train/total', loss_dict["r1"].item(), number)
                writer.add_scalar('Loss/train/total', loss_dict["path"].item(), number)
                writer.add_scalar('Loss/train/d', loss_dict["d"].item(), number)
                writer.add_scalar('Loss/train/g2', loss_dict["g"].item(), number)


            if ((epoch + 1) * batch_idx) % 50  == 0:
                state = {'Gt': Gt.state_dict()}
                torch.save(state, '/home/ai/project/agile_stage2/outcome/model_save8/Gt' +  str(number) +'.pth')


train()
