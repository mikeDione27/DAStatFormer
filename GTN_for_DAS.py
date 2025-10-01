# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:06:36 2025

@author: michel
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

#coding = UTF-8


from scipy.io import loadmat



import datetime
from sklearn import svm, preprocessing
# from get_das_data import get_das_data,get_stats_features
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns

# from transformer import ViT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
from transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
# from mytest.gather.main import draw

setup_seed(30)  # Set random seed
reslut_figure_path = r'D:\Michel\DAStatFormer\results_figure'  # Path to save result figures


path = r'D:\Michel\DAStatFormer\DAS_DataNorm_all_domains_for_GTN.mat' #r'E:\Michel\GTN-master\Gated Transformer 论文IJCAI版\DAS_Data_for_GTN_12_24.mat' 
# path = r"E:\Michel\GTN-master\Gated Transformer 论文IJCAI版\DAS_Data_Time_for_GTN_24_11.mat"
test_interval = 5  # Test interval in epochs
draw_key = 1  # Save visualizations only if epoch >= draw_key
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # Extract file name

# Hyperparameter settings
EPOCH = 200
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Select device: CPU or GPU
print(f'Using device: {DEVICE}')

d_model = 256 # 512
d_hidden = 128
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True  # Use positional encoding in one of the towers (score=pe)
mask = True  # Use input masking in one of the towers (score=input)
optimizer_name = 'Adagrad'  # Choose optimizer



# Load dataset
train_dataset = MyDataset(path, 'train')
test_dataset = MyDataset(path, 'test')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# st_dataloader = DataLoader(dataset=Xtest_gtn, batch_size=BATCH_SIZE, shuffle=False)

# Extract dataset dimensions
DATA_LEN = train_dataset.train_len        # Number of training samples
d_input = train_dataset.input_len         # Number of time steps
d_channel = train_dataset.channel_len     # Number of input features (channels)
d_output = train_dataset.output_len       # Number of output classes


# # Build the Transformer model


net = Transformer(
    d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
    q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE
).to(DEVICE)


# Define the loss function (cross-entropy)
loss_function = Myloss()


# Choose optimizer
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Lists to track metrics
correct_on_train = []
correct_on_test = []
loss_list = []
time_cost = 0

# Test function
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, *_ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        accuracy = round((100 * correct / total), 2)
        if flag == 'test_set':
            correct_on_test.append(accuracy)
        elif flag == 'train_set':
            correct_on_train.append(accuracy)
        print(f'Accuracy on {flag}: {accuracy:.2f} %')
        return accuracy

# Training function

save_path = r"D:\Michel\DAStatFormer\results_figure"



def evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pre, *_ = model(x, 'test')
            preds = y_pre.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    C = confusion_matrix(all_labels, all_preds)
    df = pd.DataFrame(C)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, fmt='g', annot=True, cmap='Reds',
                xticklabels=class_labels,
                yticklabels=class_labels,
                annot_kws={"size": 12})
    
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(rotation=0,  fontsize=12)
    plt.yticks(rotation=90, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f'{save_path}/{file_name}_Conf_Max.png')  #(save_path+"/DAStatFormer_Time_domain_confusion_matrix.jpg") 
    plt.show()
    plt.close()
    
    # === MÉTRIQUES CALCULÉES ===
    print("\n===== Performance Metrics =====")
    acc = np.trace(C) / np.sum(C)
    print('Accuracy: %.4f' % acc)
    NAR = (np.sum(C[0]) - C[0][0]) / np.sum(C[:, 1:])
    print('NAR: %.4f' % NAR)
    FNR = (np.sum(C[:, 0]) - C[0][0]) / np.sum(C[1:])
    print('FNR: %.4f' % FNR)
    column_sum = np.sum(C, axis=0)
    row_sum = np.sum(C, axis=1)
    print('Column sums:', column_sum)
    print('Row sums:', row_sum)
    
    for i in range(len(class_labels)):
        TP = C[i][i]
        precision = TP / column_sum[i] if column_sum[i] != 0 else 0.0
        recall = TP / row_sum[i] if row_sum[i] != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        print(f'Precision_{i}: {precision:.3f}')
        print(f'Recall_{i}:    {recall:.3f}')
        print(f'F1_{i}:        {f1:.3f}')



train_dataloader


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔍 Total parameters       : {total_params:,}")
    print(f"🧠 Trainable parameters   : {trainable_params:,}")
    print(f"💾 Estimated model size   : {trainable_params * 4 / 1024 ** 2:.2f} MB (float32)\n")

def train():
    print("\n📊 ==== Model Summary ====")
    count_parameters(net)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # Reset all peak memory stats

    net.train()
    max_accuracy = 0
    begin = time()

    for epoch in range(EPOCH):
        net.train()
        epoch_loss = 0.0
        batch_count = 0

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch") as tepoch:
            for x, y in tepoch:
                optimizer.zero_grad()
                y_pre, *_ = net(x.to(DEVICE), 'train')
                loss = criterion(y_pre, y.to(DEVICE))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Mémoire RAM et GPU en temps réel
                import psutil
                ram_usage = psutil.virtual_memory().used / 1024 ** 3  # GB
                postfix = {"loss": f"{loss.item():.4f}", "RAM": f"{ram_usage:.2f}GB"}

                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated(DEVICE) / 1024 ** 2
                    reserv = torch.cuda.memory_reserved(DEVICE) / 1024 ** 2
                    postfix.update({"GPU_alloc": f"{alloc:.1f}MB", "GPU_resv": f"{reserv:.1f}MB"})

                tepoch.set_postfix(postfix)

        mean_loss = epoch_loss / batch_count
        loss_list.append(mean_loss)
        print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")

        # Test tous les X epochs
        if (epoch + 1) % test_interval == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f"Max Accuracy so far — Test: {max(correct_on_test)}% | Train: {max(correct_on_train)}%")

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl')

    # Optionnel : renommer le meilleur modèle
    try:
        os.rename(f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl',
                  f'D:/Michel/DAStatFormer/saved_model/{file_name} {max_accuracy:.2f}% batch={BATCH_SIZE}.pkl')
    except Exception as e:
        print(f"[⚠️ os.rename failed] {e}")

    end = time()
    time_cost = round((end - begin) / 60, 2)
    print(f"\n⏱️ Training completed in {time_cost} min.")

    # Mémoire GPU finale
    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
        max_reserv = torch.cuda.max_memory_reserved(DEVICE) / 1024 ** 2
        print(f"📈 Max GPU Memory Allocated: {max_alloc:.2f} MB")
        print(f"📦 Max GPU Memory Reserved : {max_reserv:.2f} MB")

    evaluate_and_plot_confusion_matrix(
        model=net,
        dataloader=test_dataloader,
        device=DEVICE,
        class_labels=['background', 'digging', 'knocking', 'watering', 'shaking', 'walking'],
    )

    result_visualization(
        loss_list=loss_list,
        correct_on_test=correct_on_test,
        correct_on_train=correct_on_train,
        test_interval=test_interval,
        d_model=d_model, q=q, v=v, h=h, N=N,
        dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
        time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key,
        reslut_figure_path=reslut_figure_path, file_name=file_name,
        optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask
    )

from pathlib import Path
import matplotlib.pyplot as plt

def plot_curve(title, xlabel, ylabel, pic_file, curve1, curve2=None, legend=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(curve1)
    if curve2 is not None:
        plt.plot(curve2)
        if legend:
            plt.legend(legend, loc='best')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    Path(pic_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pic_file, dpi=160)
    plt.close()

def evaluate_loss_acc(model, dataloader, device, desc="valid"):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, *_ = model(x, 'test')
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)
    mean_loss = total_loss / max(total_count, 1)
    acc = 100.0 * total_correct / max(total_count, 1)
    print(f"{desc}: loss={mean_loss:.4f} | acc={acc:.2f}%")
    return mean_loss, acc

def train_dynamic(
    epochs=EPOCH, 
    base_lr=1e-4, 
    ckpt_dir="D:/Michel/DAStatFormer/checkpoints",
    fig_dir="D:/Michel/DAStatFormer/result_figures_wind_data",
    test_interval=1,   # on valide à chaque epoch (comme ton code UNet)
    save_every=10     # sauvegarde périodique
):
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(fig_dir);   fig_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    # Comme ton UNet: scheduler dynamique piloté par la loss de validation (mode='min')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-16, verbose=True
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_losses, valid_losses, train_accs, valid_accs, lrs = [], [], [], [], []
    best_valid_loss = float("inf")
    best_valid_acc  = -1.0

    print('ready to train (dynamic LR on validation loss)')
    for epoch in range(epochs):
        # ---------- TRAIN ----------
        net.train()
        batch_losses = []
        correct, total = 0, 0

        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, *_ = net(x, 'train')
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        running_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        running_train_acc  = 100.0 * correct / max(total, 1)
        train_losses.append(running_train_loss)
        train_accs.append(running_train_acc)

        # ---------- VALID ----------
        do_validate = ((epoch + 1) % test_interval == 0) or ((epoch + 1) == epochs)
        if do_validate:
            valid_loss, valid_acc = evaluate_loss_acc(net, test_dataloader, DEVICE, desc="valid")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            # Step du scheduler comme ton UNet (sur la VALID LOSS)
            scheduler.step(valid_loss)

            # Checkpoints “best on valid loss”
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                    ckpt_dir / "best_min_valid_loss.pth"
                )

            # Checkpoints “best on valid acc”
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(
                    {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                    ckpt_dir / "best_max_valid_acc.pth"
                )

        # LR courant (log)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        # Sauvegarde périodique
        if (epoch + 1) % save_every == 0:
            torch.save(
                {"net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                ckpt_dir / f"transformer_epoch_{epoch+1:03d}.pth"
            )

        print(f"epoch {epoch+1:3d}/{epochs:3d} | "
              f"train loss={running_train_loss:.4f} acc={running_train_acc:.2f}% | "
              f"valid loss={(valid_losses[-1] if valid_losses else float('nan')):.4f} "
              f"valid acc={(valid_accs[-1] if valid_accs else float('nan')):.2f}% | "
              f"lr={current_lr:.2e}")

    # ---------- Courbes à la fin ----------
    plot_curve("Transformer Loss (Train vs Valid)", "Epoch", "Loss",
               str(fig_dir / "loss_train_valid.png"),
               train_losses, valid_losses if valid_losses else None, legend=["train", "valid"])
    plot_curve("Transformer Learning Rate", "Epoch", "LR",
               str(fig_dir / "learning_rate.png"), lrs)

    print("training finished with dynamic LR scheduling")

# evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels)

if __name__ == '__main__':
    # train_dynamic()
    train()

