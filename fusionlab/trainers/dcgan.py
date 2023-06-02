# Ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import numpy as np
from fusionlab.layers import ConvND, BatchNorm, ConvT

class Generator(nn.Module):
    def __init__(self, c_in, c_out, dim_g, spatial_dims=2):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            ConvT(spatial_dims, c_in, dim_g * 8, 4, 1, 0, bias=False),
            BatchNorm(spatial_dims,dim_g * 8),
            nn.ReLU(True),
            # state size. (dim_g*8) x 4 x 4
            ConvT(spatial_dims, dim_g * 8, dim_g * 4, 4, 2, 1, bias=False),
            BatchNorm(spatial_dims,dim_g * 4),
            nn.ReLU(True),
            # state size. (dim_g*4) x 8 x 8
            ConvT(spatial_dims, dim_g * 4, dim_g * 2, 4, 2, 1, bias=False),
            BatchNorm(spatial_dims, dim_g * 2),
            nn.ReLU(True),
            # state size. (dim_g*2) x 16 x 16
            ConvT(spatial_dims, dim_g * 2, dim_g, 4, 2, 1, bias=False),
            BatchNorm(spatial_dims, dim_g),
            nn.ReLU(True),
            # state size. (dim_g) x 32 x 32
            ConvT(spatial_dims, dim_g, c_out, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, c_in, dim_d, spatial_dims=2):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            ConvND(spatial_dims, c_in, dim_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            ConvND(spatial_dims, dim_d, dim_d * 2, 4, 2, 1, bias=False),
            BatchNorm(spatial_dims, dim_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            ConvND(spatial_dims, dim_d * 2, dim_d * 4, 4, 2, 1, bias=False),
            BatchNorm(spatial_dims, dim_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            ConvND(spatial_dims, dim_d * 4, dim_d * 8, 4, 2, 1, bias=False),
            BatchNorm(spatial_dims, dim_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            ConvND(spatial_dims, dim_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class DCGANTrainer:
    def __init__(self, generator, discriminator, optim_g, optim_d, loss_fn, device, dim_z, spatial_dims=2):
        self.g = generator.to(device)
        self.d = discriminator.to(device)
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.loss_fn = loss_fn
        self.dim_z = dim_z
        self.device = device
        self.fixed_noise = None
        self.spatial_dims = spatial_dims

    def train_step(self, real_x):

        bs = real_x.size(0)
        label_real = torch.full((bs,), 1., dtype=torch.float, device=self.device)
        label_fake = torch.full((bs,), 0., dtype=torch.float, device=self.device)
        if self.spatial_dims==1:
            noise_x = torch.rand(bs, self.dim_z, 1, device=self.device)
        elif self.spatial_dims==2:
            noise_x = torch.rand(bs, self.dim_z, 1, 1, device=self.device)
        elif self.spatial_dims==3:
            noise_x = torch.rand(bs, self.dim_z, 1, 1, 1, device=self.device)

        # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        self.d.zero_grad()

        # real -> D
        output_d_real = self.d(real_x).view(-1)
        # noise -> G -> D
        fake = self.g(noise_x)
        output_g_d = self.d(fake.detach()).view(-1)

        # Discriminator loss on real and fake batch
        loss_d_real = self.loss_fn(output_d_real, label_real)
        loss_d_fake = self.loss_fn(output_g_d, label_fake)
        loss_d = loss_d_real.mean() + loss_d_fake.mean()

        loss_d_real.backward()
        loss_d_fake.backward()
        self.optim_d.step()

        # Train Generator: minimize log(1-D(G(z)))
        self.g.zero_grad()
        output_g = self.d(fake).view(-1)
        loss_g = self.loss_fn(output_g, label_real)

        loss_g.backward()
        optimizerG.step()

        log = {'loss_d': loss_d.item(), 'loss_g': loss_g.item()}
        return log

    def fit(self, dataloader, epochs):
        log = {'loss_d': [], 'loss_g': []}
        if self.spatial_dims==1:
            fixed_noise = torch.rand(32, self.dim_z, 1, device=self.device)
        elif self.spatial_dims==2:
            fixed_noise = torch.rand(32, self.dim_z, 1, 1, device=self.device)
        elif self.spatial_dims==3:
            fixed_noise = torch.rand(32, self.dim_z, 1, 1, 1, device=self.device)
        for epoch in tqdm(range(epochs)):
            epoch_log = {'loss_d': [], 'loss_g': []}
            for i, (x, _) in enumerate(tqdm(dataloader)):
                real_x = x.to(self.device)
                step_log = self.train_step(real_x)
                [epoch_log[k].append(v) for k, v in step_log.items()]
            [log[k].append(np.mean(v)) for k, v in epoch_log.items()]
            print(f'''[{epoch}/{epochs}] loss_d: {log['loss_d'][-1]} loss_g: {log['loss_g'][-1]}''')
        return log


if __name__ == '__main__':
    BS = 64
    image_size = 64
    img_channel = 1

    dim_z = 100
    dim_g = 64
    dim_d = 64

    lr = 0.0002
    beta1 = 0.5

    dataset = torchvision.datasets.MNIST(
        root="",
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]),
        download=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BS,
                                             shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_g = Generator(dim_z, img_channel, dim_g, spatial_dims=2)
    model_d = Discriminator(img_channel, dim_d, spatial_dims=2)

    criterion = nn.BCELoss()

    optimizerD = torch.optim.Adam(model_d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(model_g.parameters(), lr=lr, betas=(beta1, 0.999))

    # DCGAN trainer
    trainer = DCGANTrainer(model_g, model_d, optimizerG, optimizerD, criterion, device, dim_z, spatial_dims=2)
    log = trainer.fit(dataloader, 3)
