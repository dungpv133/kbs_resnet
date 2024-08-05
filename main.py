import pandas as pd
import torch
import argparse
from torch import nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from dataset import SoundDS
from basic_model import AudioClassifier
import numpy as np
from resnet import ResNetCustom

# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
def prepara_data():
    data_path = '/kaggle/input/kbs-clean/data_cleaned/data_cleaned/'

    # Read metadata file
    # metadata_file = '/content/Audio_Deep_Learning/Audio_Classification/simple_df.csv'
    # kaggle
    metadata_file = '/kaggle/input/kbs-clean/temp_cleaned.csv'
    metadata_file = '/kaggle/input/kbs-clean/label_test.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    # df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    myds = SoundDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    return (train_dl, val_dl)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ----------------------------
# Training Loop
# ----------------------------
def training(train_dl, num_epochs, test_dl, args, device):
    # Tensorboard 
    writer = SummaryWriter()

    # Create the model and put it on the GPU if available
    model = nn.DataParallel(ResNetCustom(num_classes = 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('/kaggle/input/kbs-clean/best_resnet_custom.pt'))
    model = model.to(device)

    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    lr = args.lr
    max_lr = args.maxlr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
    #                                             steps_per_epoch=int(len(train_dl)),
    #                                             epochs=num_epochs,
    #                                             anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience= 8, min_lr =1e-4, verbose=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Repeat for each epoch
    num_epochs = args.epochs
    best_acc = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = np.squeeze(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            #if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        avg_acc = correct_prediction/total_prediction
        current_lr = get_lr(optimizer)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Acc/train", avg_acc, epoch)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}, Current LR: {current_lr}')

        acc = inference(model, test_dl, device)
        if(acc > best_acc):
            best_acc = acc
            print(f"Save best model with accuracy: {best_acc}")
            torch.save(model.state_dict(), 'model_command.pt') 
    
    print('Finished Training')

# ----------------------------
# Inference
# ----------------------------
def inference (model, test_dl, device):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        with open('command_detection.txt', 'a') as file:
            for data in test_dl:
                # Get the input features and target labels, and put them on the GPU
                inputs, labels, filename = data[0].to(device), data[1].to(device), data[2].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = model(inputs)

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                file.write(filename.replace('.wav', '') + " " + str(prediction) + '\n')
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f"Validation:")
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    return acc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default='test')
    ap.add_argument("--epochs", type = int, default='50')
    ap.add_argument("--lr", type = float, default='0.001')
    ap.add_argument("--maxlr", type = float, default='0.001')
    # args = vars(ap.parse_args)
    args = ap.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = prepara_data()
    mode = args.mode
    if mode == 'train':
        # Run training model
        training(train_dl, 50, test_dl, args, device)
    else:
        # Run inference on trained model with the validation set load best model weights
        # Load trained/saved model
        model_inf = nn.DataParallel(ResNetCustom(num_classes = 2))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_inf = model_inf.to(device)
        model_inf.load_state_dict(torch.load('/kaggle/working/kbs_resnet/model.pt'))
        model_inf.eval()

        # Perform inference
        inference(model_inf, test_dl, device)

