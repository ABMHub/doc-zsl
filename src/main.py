from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER

config = Config(
  batch_size=64
)
model = CCT()
model = SiameseModel(model)

csv_path = "./tobacco_df.csv"

train_loader = DocDataset(csv_path, True, False)
val_loader = DocDataset(csv_path, False, False)

train_loader = ContrastivePairLoader(train_loader)
val_loader = ContrastivePairLoader(val_loader)

train_loader = DataLoader(train_loader, config.batch_size, shuffle=False)
val_loader = DataLoader(val_loader, config.batch_size, shuffle=False)

log = Log()
log.create_metric("eer", EER(), True)
log.create_metric("eer", EER(), False)

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log
)