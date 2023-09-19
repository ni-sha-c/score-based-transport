import torch
import torch.nn as nn
import tqdm as progressbar
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
from matplotlib.pyplot import *
class v_mlp(nn.Module):
    def __init__(self, dim=2):
        super(v_mlp, self).__init__()
        self.relu = nn.Sigmoid()
        n = 128
        self.fc1 = nn.Linear(dim, n)  
        self.fc2 = nn.Linear(n, 2*n)
        self.fc3 = nn.Linear(2*n, 2*n)
        self.fc4 = nn.Linear(2*n, dim)
    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        x = self.fc4(self.relu(self.fc3(x)))
        return x

class v_fc(nn.Module):
    def __init__(self, dim=2):
        super(v_fc, self).__init__()
        self.relu = nn.Sigmoid()
        n = 8
        self.fc1 = nn.Linear(dim, 64)  
        self.conv1 = nn.Conv2d(1, n, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 8x8x4
        self.fc2 = nn.Linear(16*n, dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.view(1,8,8)
        x = self.pool1(self.relu(self.conv1(x)))
        x = x.flatten()
        x = self.fc2(x)
        return x


class v_lstm(nn.Module):
    def __init__(self, dim=2):
        super(v_lstm, self).__init__()
        self.relu = nn.Sigmoid()
        self.fc1 = nn.Linear(dim, 64)  
        n = 4
        # Encoder
        self.e11 = nn.Conv2d(1, n, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(n, n, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 8x8x4

        self.e21 = nn.Conv2d(n, 2*n, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(2*n, 2*n, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 16x4x1
        self.e31 = nn.Conv2d(2*n, 4*n, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(4*n, 4*n, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x2x1
        self.e41 = nn.Conv2d(4*n, 8*n, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(8*n, 8*n, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(8*n, 4*n, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(8*n, 4*n, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(4*n, 4*n, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(4*n, 2*n, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(4*n, 2*n, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(2*n, 2*n, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(2*n, n, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(2*n, n, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(n, n, kernel_size=3, padding=1)

        # out conv
        self.outconv = nn.Conv2d(n, 1, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(64, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(1, 8, 8)
        # encoder
        xe1 = self.relu(self.e12(self.relu(self.e11(x))))
        x = self.pool1(xe1)
        xe2 = self.relu(self.e22(self.relu(self.e21(x))))
        x = self.pool2(xe2)
        xe3 = self.relu(self.e32(self.relu(self.e31(x))))
        x = self.pool3(xe3)
        x = self.relu(self.e42(self.relu(self.e41(x))))
        # decoder
        x = self.upconv1(x)
        xu1 = torch.cat([x, xe3], dim=0)
        x = self.relu(self.d12(self.relu(self.d11(xu1))))
        x = self.upconv2(x)
        xu2 = torch.cat([x, xe2], dim=0)
        x = self.relu(self.d22(self.relu(self.d21(xu2))))
        x = self.upconv3(x)
        xu3 = torch.cat([x, xe1], dim=0)
        x = self.relu(self.d32(self.relu(self.d31(xu3))))
        x = self.outconv(x)
        x = x.flatten()
        x = self.fc2(x)
        return x

class v_unet(nn.Module):
    def __init__(self, dim=2):
        super(v_unet, self).__init__()
        self.relu = nn.Sigmoid()
        self.fc1 = nn.Linear(dim, 8)  
        n = 4
        # Encoder
        self.e11 = nn.Conv2d(1, n, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(n, n, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 8x8x4

        self.e21 = nn.Conv2d(n, 2*n, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(2*n, 2*n, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 16x4x1
        self.e31 = nn.Conv2d(2*n, 4*n, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(4*n, 4*n, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x2x1
        self.e41 = nn.Conv2d(4*n, 8*n, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(8*n, 8*n, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(8*n, 4*n, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(8*n, 4*n, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(4*n, 4*n, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(4*n, 2*n, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(4*n, 2*n, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(2*n, 2*n, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(2*n, n, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(2*n, n, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(n, n, kernel_size=3, padding=1)

        # out conv
        self.outconv = nn.Conv2d(n, 1, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(64, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(1, 8, 8)
        # encoder
        xe1 = self.relu(self.e12(self.relu(self.e11(x))))
        x = self.pool1(xe1)
        xe2 = self.relu(self.e22(self.relu(self.e21(x))))
        x = self.pool2(xe2)
        xe3 = self.relu(self.e32(self.relu(self.e31(x))))
        x = self.pool3(xe3)
        x = self.relu(self.e42(self.relu(self.e41(x))))
        # decoder
        x = self.upconv1(x)
        xu1 = torch.cat([x, xe3], dim=0)
        x = self.relu(self.d12(self.relu(self.d11(xu1))))
        x = self.upconv2(x)
        xu2 = torch.cat([x, xe2], dim=0)
        x = self.relu(self.d22(self.relu(self.d21(xu2))))
        x = self.upconv3(x)
        xu3 = torch.cat([x, xe1], dim=0)
        x = self.relu(self.d32(self.relu(self.d31(xu3))))
        x = self.outconv(x)
        x = x.flatten()
        x = self.fc2(x)
        return x

def pde(model, x):
    x.requires_grad_(True)
    y = model(x)
    #dvx_dx = torch.autograd.grad(y[0], x, retain_graph=True)
    dvx_dx = y[0].unsqueeze(0)
    #dvy_dx = torch.autograd.grad(y[1], x, retain_graph=True)
    #d2vx_dx2 = torch.autograd.grad(dvx_dx, x, create_graph=True)
    #out = sum(dvx_dx[0])
    return dvx_dx

def pde_rhs(x):
    with torch.no_grad ():
        rhs = torch.pi * torch.sin(torch.pi * (x[0] + x[1]))
    return rhs

def create_dataset(m=100,xmin=torch.tensor([0.0, 0.0]), xmax=torch.tensor([1.0, 1.0])):
    class PDEDataset(Dataset):
        def __init__(self, m, xmin, xmax):
            self.x = xmin + (xmax-xmin)*torch.rand(m, 2)
            self.y = torch.zeros(m, 2)
            self.m = m
            for i in range(m):
                self.y[i] = torch.tensor([pde_rhs(self.x[i]),0.0])
        def __len__(self):
            return self.m
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
    dataset = PDEDataset(m,xmin,xmax)
    return dataset

def train_pde(model, pde, xmin=torch.tensor([0.0,0.0]), xmax=torch.tensor([1.0,1.0]), m=100, n=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    dataset = create_dataset(m, xmin, xmax)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    print("Training the model...")
    with progressbar.tqdm (range(n), unit="epoch") as pbar:
        for epoch in range(n):
            optimizer.zero_grad()
            for t, (x, y) in enumerate(dataloader):
                x_g = x.to(args.device)
                y_g = y.to(args.device)
                output = torch.zeros(args.batchsize, 2, requires_grad=True)
                output = output.to(args.device)
                for i, x_i in enumerate(x_g):
                    output[i,0] = pde(model, x_i)
                loss = torch.nn.functional.mse_loss(output, y_g)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 25 == 0:
                pbar.write(f"Epoch {epoch}: loss={loss.item():.3e}")
            pbar.update()
    return model
def visualize(model, xmin=torch.tensor([0.0,0.0]), xmax=torch.tensor([1.0,1.0]), n_gr=100):
    x = torch.linspace(xmin[0], xmax[0], n_gr)
    y = torch.linspace(xmin[1], xmax[1], n_gr)
    X, Y = torch.meshgrid(x, y)
    Z_nn = torch.zeros((n_gr, n_gr))
    Z = torch.zeros((n_gr, n_gr))
    for i in range(n_gr):
        for j in range(n_gr):
            inp = torch.tensor([X[i,j], Y[i,j]]).to(args.device)
            Z_nn[i,j] = model(inp)[0]
            #Z[i,j] = torch.sin(torch.pi * X[i,j]) * torch.sin(torch.pi * Y[i,j])
            Z[i,j] = torch.pi * torch.sin(torch.pi * (X[i,j] + Y[i,j]))
    fig, ax = subplots()
    CS = ax.contourf(X, Y, Z_nn.cpu().detach().numpy())
    cbar = fig.colorbar(CS, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title('NN Solution of the PDE',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()

    fig, ax = subplots()
    CS = ax.contourf(X, Y, Z)
    cbar = fig.colorbar(CS, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title('Exact Solution of the PDE',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    tight_layout()
    show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--m", type=int, default=100)
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args) 

    model = v_fc(dim=2)
    model = model.to(args.device)
    xmin = torch.tensor([0.0, 0.0])
    xmax = torch.tensor([1.0, 1.0])
    m = args.m
    n = args.epoch
    model = train_pde(model, pde, xmin, xmax, m, n)
    print("Training is done!")
    print("Visualizing ...")
    visualize(model, xmin, xmax, 100)


