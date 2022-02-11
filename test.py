import torch
from stylegan2.stylegan2_pytorch.model import Generator, MLP
from torchvision import utils
import cv2
import torchvision.transforms as transforms
from AgileGAN_main.lib.normal_image import  Normal_Image
import argparse
import numpy as np
import torch.nn.functional as F
from torch import nn
from AgileGAN_main.lib.encoder.stylegan2.model import EqualLinear
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from AgileGAN_main.lib.encoder.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE


class SubBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(SubBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class VAEStyleEncoder(Module):
    def __init__(self, num_layers, opts=None):
        super(VAEStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152]
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = SubBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = SubBlock(512, 512, 32)
            else:
                style = SubBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)


        self.fc_mu = nn.Linear(512, 512)
        self.fc_var = nn.Linear(512, 512)

        self.fc_mu.weight.data.fill_(0)
        self.fc_mu.bias.data.fill_(0)

        self.fc_var.weight.data.fill_(0)
        self.fc_var.bias.data.fill_(0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        out = torch.stack(latents, dim=1)

        mu = self.fc_mu(out)
        logvar = self.fc_var(out)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, logvar, mu


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class tester():
    def __init__(self):
        self.device0 = "cuda: 0"

        d_path = "/home/ai/project/agile_stage2/Decoders/Gt1095.pth"
        checkpoint_d = torch.load(d_path)
        self.D = Generator(output_size, 512, 8).to(device0)
        D.load_state_dict(checkpoint_d["Gt"])

        e_path = 'AgileGAN_main/pretrain/encoder.pt'
        checkpoint_e = torch.load(e_path)
        self.E = VAEStyleEncoder.to(device0)
        self.E.load_state_dict(get_keys(checkpoint_e, 'encoder'), strict=True)

        mlp_path = "/home/ai/project/agile3/pretrained_models/stylegan2-ffhq-config-f.pt"
        checkpoint_mlp = torch.load(mlp_path)
        self.mlp =MLP(1024, 512, 8).to(device0)
        self.mlp.load_state_dict(checkpoint_mlp["g_ema"], strict=False)

        self.normal = Normal_Image()
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def run(self, img_path):
        img = cv2.imread(img_path)
        if (img is not None):
            img = self.normal.run(img)
            img = img.convert("RGB")
            transformed_image = self.transforms(img)
            _, _, mu = self.E(transformed_image.unsqueeze(0).to("cuda").float())
            #latent = [self.pspnet.decoder.style(s) for s in mu]
            #latent = [torch.stack(latent, dim=0)]
            lalent = mlp(mu)
            latent = torch.stack(latent, dim=0).to(device1)
            fake_img_A = self.G(latent, input_is_latent=True)
            utils.save_image(
                fake_img_A,
                "output.jpg",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        else:
            print('img is None')
            return 0

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='examples/29899.jpg')
input_opts = parser.parse_args()

T=Tester()
print("start to infer.")
T.run(input_opts.path)
print("done.")


