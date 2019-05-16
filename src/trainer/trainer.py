import torch.nn.functional as F

class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(self, num_epoch):

        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        self.model.train()

        for epoch in num_epoch:
           self.optimizer.zero_grad()
           out = self.model(data)
           loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
           loss.backward()
           self.optimizer.step()

           print(loss)
