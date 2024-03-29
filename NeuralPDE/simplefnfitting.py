import torch
import torch.nn as nn
import tqdm as progressbar
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
from matplotlib.pyplot import *
class v_fc(nn.Module):
    def __init__(self, dim=2):
        super(v_fc, self).__init__()
        self.relu = nn.Sigmoid()
        n_conv_o = 16
        n_conv_i = 2
        n_fc = 128
        self.n_conv_o = n_conv_o
        self.n_conv_i = n_conv_i
        self.n_fc = n_fc
        self.fc1 = nn.Linear(dim, n_fc)  
        self.conv1 = nn.Conv2d(n_conv_i, n_conv_o, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 8x8x4
        self.fc2 = nn.Linear(n_fc//n_conv_i//4*n_conv_o, dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.view(self.n_conv_i,self.n_fc//self.n_conv_i//4,4)
        x = self.pool1(self.relu(self.conv1(x)))
        x = x.flatten()
        x = self.fc2(x)
        return x

def pde(model, x):
    x.requires_grad_(True)
    y = model(x)
    #dvx_dx = torch.autograd.grad(y[0], x, retain_graph=True)
    dvx_dx = y[0]
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--m", type=int, default=300)
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


