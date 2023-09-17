import torch
import torch.nn as nn
import tqdm as progressbar
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
    dvx_dx = torch.autograd.grad(y[0], x, create_graph=True)
    dvy_dx = torch.autograd.grad(y[1], x, create_graph=True)
    #d2vx_dx2 = torch.autograd.grad(dvx_dx, x, create_graph=True)
    out = sum(dvx_dx[0])
    return out

def pde_rhs(x):
    with torch.no_grad ():
        rhs = torch.pi * torch.sin(torch.pi * (x[0] + x[1]))
    return rhs

def train_pde(model, pde, pde_rhs, m=100, n=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Training the model...")
    with progressbar.tqdm (range(n), unit="epoch") as pbar:
        for epoch in range(n):
            optimizer.zero_grad()
            loss = 0.0
            for i in range(m):
                x = torch.randn(2)
                pdex = pde(model, x)
                pdex_rhs = pde_rhs(x)
                loss += (pdex - pdex_rhs)**2
            loss.backward()
            optimizer.step()
            pbar.write(f"Epoch {epoch}: loss={loss.item():.4e}")
            pbar.update()
    return model
# Create an instance of the model
#model = v_dnn(dim=2)

# Print the model architecture
#print(model)

# evaluate the model
#model.eval()
x = torch.randn(2)
model = v_dnn(dim=2)
model = train_pde(model, pde, pde_rhs, m=100)

#print("True derivative: ", torch.cat((2*x.unsqueeze(0), torch.zeros(2).unsqueeze(0)), dim=0))

