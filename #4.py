class Model(torch.nn.Module):
  def __init__(self):
     super().__init__()
     self.fc=nn.Linear(30000, 30000)
  def forward(self, x):
    return self.fc(x)
x = torch.rand(30000, 30000, dtype=torch.float32, device='cuda')
model=Model()
model.to(device)
model(x)
