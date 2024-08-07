import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import logging
import json

from dataset.Dataset import WheelDataset
from model.Model import Model
from during_train import train_one_epoch, test_one_epoch

with open("./config/WHTextNet.json", 'r') as f:
    cfg = json.load(f)
    print(cfg)
# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(filename="./log/"+timestamp+".log",level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info(cfg['network'])


train_data = WheelDataset(cfg)
print("training set has {} instances".format(len(train_data)))

train_dataloader = DataLoader(train_data, batch_size=cfg['train']['batch'], shuffle=True)
valid_dataloader = DataLoader(train_data, batch_size=cfg['train']['batch'], shuffle=False)

model = Model()
print(f"Number of Parameters:{sum(p.numel() for p in model.parameters())}")
model.cuda()
# if cfg['train']['resume'] is not None:
#     model.load_state_dict(torch.load(cfg['train']['resume']))
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
loss_fn.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = cfg['train']['max_epoch']
best_vloss = 1000000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, train_dataloader, optimizer, model, loss_fn)
    scheduler.step()

    avg_vloss, correct = test_one_epoch(valid_dataloader, model, loss_fn)
    logging.info(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_loss}, Train Accuracy: {correct}')

    # Track the best performance, and save the model's state
    if epoch_number % cfg['train']['save_per_epoch'] == 0:
        model_path = cfg['train']['save_pth_dir'] + 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    print("\n")

    epoch_number += 1

logging.shutdown()
