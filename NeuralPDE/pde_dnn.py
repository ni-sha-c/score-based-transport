import torch
import torch.nn as nn
import tqdm as progressbar
from torch.utils.data import Dataset, DataLoader
import argparse
class v_dnn(nn.Module):
    def __init__(self, dim=2):
        super(v_dnn, self).__init__()
        self.fc1 = nn.Linear(dim, 512)  
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(512, 512)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(512, dim)

    def forward(self, x):
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.fc3(x)
        return x

def pde(model, x):
    x.requires_grad_(True)
    y = model(x)
    print("NN evaluated at %f is %f" % (x[0], y[0]))
    dvx_dx = torch.autograd.grad(y[0], x)
    dvy_dx = torch.autograd.grad(y[1], x)
    #d2vx_dx2 = torch.autograd.grad(dvx_dx, x, create_graph=True)
    out = sum(dvx_dx[0])
    return out

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
                self.y[i] = pde_rhs(self.x[i])
        def __len__(self):
            return self.m
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
    dataset = PDEDataset(m,xmin,xmax)
    return dataset

def train_pde(model, pde, xmin=torch.tensor([0.0,0.0]), xmax=torch.tensor([1.0,1.0]), m=100, n=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dataset = create_dataset(m, xmin, xmax)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    print("Training the model...")
    with progressbar.tqdm (range(n), unit="epoch") as pbar:
        for epoch in range(n):
            optimizer.zero_grad()
            loss = 0.0
            for t, (x, y) in enumerate(dataloader):
                x_g = x.to(args.device)
                y_g = y.to(args.device)
                output = torch.zeros(args.batchsize, 2)
                output = output.to(args.device)
                for i, x_i in enumerate(x_g):
                    output[i] = pde(model, x_i)
                loss += torch.nn.functional.mse_loss(output, y_g)
                loss.backward()
                optimizer.step()
            pbar.write(f"Epoch {epoch}: loss={loss.item():.4e}")
            pbar.update()
    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args) 

    model = v_dnn(dim=2)
    model = model.to(args.device)
    xmin = torch.tensor([0.0, 0.0])
    xmax = torch.tensor([1.0, 1.0])
    m = 100
    n = args.epoch
    model = train_pde(model, pde, xmin, xmax, m, n)


# Create an instance of the model
#model = v_dnn(dim=2)

# Print the model architecture
#print(model)

# evaluate the model
#model.eval()
#x = torch.randn(2)
#model = v_dnn(dim=2)
#model = train_pde(model, pde, pde_rhs, m=100)

#print("True derivative: ", torch.cat((2*x.unsqueeze(0), torch.zeros(2).unsqueeze(0)), dim=0))

