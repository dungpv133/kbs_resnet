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
    data_path = '/kaggle/input/kbs-clean/command_detection_test'

    # Read metadata file
    # metadata_file = '/content/Audio_Deep_Learning/Audio_Classification/simple_df.csv'
    # kaggle
    metadata_file = '/kaggle/input/kbs-clean/temp_cleaned.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    # df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    myds = SoundDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
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


# ----------------------------
# Inference
# ----------------------------
def inference (model, test_dl, device):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f"Validation:")
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    return acc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default='train')
    ap.add_argument("--epochs", type = int, default='50')
    ap.add_argument("--lr", type = float, default='0.001')
    ap.add_argument("--maxlr", type = float, default='0.001')
    # args = vars(ap.parse_args)
    args = ap.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = prepara_data()
    mode = args.mode
    
    # Run inference on trained model with the validation set load best model weights
    # Load trained/saved model
    model_inf = nn.DataParallel(ResNetCustom(num_classes = 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inf = model_inf.to(device)
    model_inf.load_state_dict(torch.load('/kaggle/working/kbs_resnet/model_command.pt'))
    model_inf.eval()

    # Perform inference
    inference(model_inf, test_dl, device)

