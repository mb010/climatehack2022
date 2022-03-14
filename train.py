import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ClimateHackDataset
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

import logging
import os
from time import gmtime



# These will change depending on my configuration
from submission.model import Model
from loss import MS_SSIMLoss

class scale():
    def __init__(self, limit):
        self.limit = limit
    def __call__(self, image, **kwargs):
        return image/self.limit

def train():
    logging.Formatter.converter = gmtime
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='out.log', format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)    
    logger.info("Initialising Training")
    EPOCHS=int(1000)
    EARLY_STOPPING_AFTER=50
    TARGET_TIME_STEPS=1
    BATCH_SIZE=8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = 64
    
    transform = A.Compose(
        [
            A.RandomCrop(imsize,imsize),
            A.Lambda(
                name='scale',
                image=scale(1023),
            )
        ],
        additional_targets={'target': 'image'}
    )
    
    # ConvLSTM
    # https://github.com/ndrplz/ConvLSTM_pytorch
    logger.info("Loading Model")
    model = Model(
        input_dim=1,
        hidden_dim=[64, 128, 128, TARGET_TIME_STEPS],
        kernel_size=(5,5),
        num_layers=4,
        batch_first=True,
        bias=True,
        return_all_layers=False
    ).to(device)
    
    #criterion = nn.MSELoss()
    criterion = MS_SSIMLoss(channels=TARGET_TIME_STEPS)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    

    logger.info("Loading Data")
    trainloader = DataLoader(
        ClimateHackDataset(
            split='train',
            transform=transform,
            target_time_steps=TARGET_TIME_STEPS
        ), 
        batch_size=BATCH_SIZE, 
        num_workers=8,
        timeout=1000,
        prefetch_factor=2,
    )
    validloader = DataLoader(
        ClimateHackDataset(
            split='val',
            transform=transform,
            target_time_steps=TARGET_TIME_STEPS
        ), 
        batch_size=BATCH_SIZE, 
        num_workers=8,
        timeout=1000,
        prefetch_factor=2,
    )
    
    ### Training loop ###
    val_loss_min = np.inf
    counter = 0
    losses = []
    logger.info("Starting training loop")
    for epoch in range(EPOCHS):
        logger.info(f"EPOCH {epoch}")
        running_loss = 0
        count = 0
        optimiser.zero_grad()

        for idx, (batch_features, batch_targets) in enumerate(trainloader):
            
            model.train()
            optimiser.zero_grad()
            layer_output, last_state  = model(batch_features.to(device))
            batch_predictions = layer_output[0][-1]
            #print(f"Sizes IN: {batch_predictions.shape}\nSizes OUT: {batch_targets[:,0].shape}")
            batch_loss = criterion(
                batch_predictions,
                batch_targets[:,:,0].to(device)
            )
            batch_loss.backward()
            optimiser.step()

            running_loss += batch_loss.item() * batch_predictions.shape[0]
            count += batch_predictions.shape[0]
            #if epoch<=3:
            #    print(f"epoch {epoch} with running_loss {running_loss}")
            logger.info(f"epoch {epoch} batch {idx}: {batch_loss:.5e}")
           
        if epoch>10:
            ### Validation step & early stopping ###
            val_loss = 0.
            for val_x, val_y in validloader:
                model.eval()
                with torch.no_grad():
                    layer_output, last_state = model(val_x.to(device))
                    val_predictions = layer_output[0][-1]
                    val_loss += criterion(
                        val_predictions,
                        val_y[:,:,0].to(device)
                    )#/val_y.shape[0]
            if val_loss <= val_loss_min:
                val_loss_min=val_loss
                best = {
                    'epoch':epoch,
                    'loss':val_loss_min,
                    'weights': model.state_dict()
                }
                counter = 0
            else:
                counter += 1
            if counter >= EARLY_STOPPING_AFTER:
                break

        losses.append(running_loss / count)
        logger.info(f"Epoch: {epoch}/{EPOCHS}; running_loss/count: {losses[-1]}; validation_loss: {val_loss}; min_validation: {counter==0}")
        if epoch in np.arange(0,1000,50):
            torch.save(model.state_dict(), f'submission/model_{epoch}.pt')
            
    np.save("./losses.npy", np.asarray(losses))
    
    torch.save(model.state_dict(), 'submission/model.pt')
    
if __name__=="__main__":
    train()