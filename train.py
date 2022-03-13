import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ClimateHackDataset
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

# These will change depending on my configuration
from submission.model import Model
from loss import MS_SSIMLoss

class scale():
    def __init__(self, limit):
        self.limit = limit
    def __call__(self, image, **kwargs):
        return image/self.limit

def train():
    EPOCHS=int(5*10e2)
    EARLY_STOPPING_AFTER=20
    TARGET_TIME_STEPS=24
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
    model = Model(
        input_dim=1,
        hidden_dim=[64, 64, 64, TARGET_TIME_STEPS],
        kernel_size=(3,3),
        num_layers=4,
        batch_first=True,
        bias=True,
        return_all_layers=False
    ).to(device)
    
    #criterion = nn.MSELoss()
    criterion = MS_SSIMLoss(channels=TARGET_TIME_STEPS)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    

    trainloader = DataLoader(
        ClimateHackDataset(
            split='train',
            transform=transform,
            target_time_steps=TARGET_TIME_STEPS
        ), 
        batch_size=BATCH_SIZE, 
        num_workers=8,
        timeout=5, 
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
        timeout=5, 
        prefetch_factor=2,
    )
    
    ### Training loop ###
    val_loss_min = np.inf
    counter = 0
    losses = []
    print(">>> TRAINING START")
    for epoch in range(EPOCHS):
        running_loss = 0
        count = 0
        optimiser.zero_grad()

        for batch_features, batch_targets in trainloader:
            
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
        print(f"Epoch: {epoch}/{EPOCHS}; running_loss/count: {losses[-1]}")
    np.save("./losses.npy", np.asarray(losses))
    
    torch.save(model.state_dict(), 'submission/model.pt')
    
    ### SAVE EXAMPLE FIGURE ### 
    val = ClimateHackDataset(
        split='val',
        transform=transform
    )

    x, y = val[10]
    print(x.shape, y.shape)
    a, b = model(torch.from_numpy(x.astype(np.float32)).unsqueeze(dim=0).to(device))

    p = b[0][0]
    p = p.squeeze().cpu().detach().numpy().squeeze()
    plt.hist(p.flatten())
    plt.show()
    print(p.shape)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 12, figsize=(20,6))

    # plot the twelve 128x128 input images
    for i, img in enumerate(x):
        ax1[i].imshow(img.squeeze(), cmap='viridis')
        ax1[i].get_xaxis().set_visible(False)
        ax1[i].get_yaxis().set_visible(False)

    # plot twelve 64x64 true output images
    for i, img in enumerate(y[:12]):
        ax2[i].imshow(img.squeeze(), cmap='viridis')
        ax2[i].get_xaxis().set_visible(False)
        ax2[i].get_yaxis().set_visible(False)

    # plot the twelve 64x64 predicted output images
    for i, img in enumerate(p[:12]):
        ax3[i].imshow(img, cmap='viridis')
        ax3[i].get_xaxis().set_visible(False)
        ax3[i].get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig('./model_out_example.png')

if __name__=="__main__":
    train()