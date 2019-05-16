from src.trainer import Trainer
from src.models import TwoLayerGCN

trainer = Trainer(TwoLayerGCN)
trainer.train(100)
