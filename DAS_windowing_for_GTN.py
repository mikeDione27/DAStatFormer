# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:53:00 2025

@author: michel
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os

class DASMatDataset(Dataset):
    def __init__(self, root_dir, window_size=200, stride=100, key='data', transform=None):
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.key = key  # nom de la variable dans le .mat (à adapter selon ton cas)

        class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for label, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.mat'):
                    full_path = os.path.join(class_path, file)
                    mat = scipy.io.loadmat(full_path)
                    if self.key not in mat:
                        raise ValueError(f"Clé '{self.key}' introuvable dans le fichier {file}")
                    data = mat[self.key]  # <- doit être shape (10000, 12)
                    if data.shape[0] < window_size:
                        continue  # ignorer si trop court
                    for i in range(0, data.shape[0] - window_size + 1, stride):
                        window = data[i:i+window_size]
                        self.samples.append(window)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = self.samples[idx]  # shape (50, 12), dtype np.float32 normalement
        y = self.labels[idx]
    
        # ✅ Normalisation globale
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    
        # ✅ Conversion en Tensors (float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)



    # def __getitem__(self, idx):
    #     x = self.samples[idx].astype(np.float32)
    #     y = self.labels[idx]
    
    #     # ⚙️ Normalisation standard par fenêtre
    #     x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)  # shape (200, 12)
    
    #     if self.transform:
    #         x = self.transform(x)
    
    #     return x, y

import torch
import os
import scipy.io
import numpy as np
from torch.utils.data import Dataset

import os
import scipy.io
import numpy as np
from torch.utils.data import Dataset

class DASMatDataset_Max(Dataset):
    def __init__(self, root_dir, window_size=200, key='data'):
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.key = key  # nom de la variable dans le .mat

        # Récupère les sous-dossiers triés (classes)
        class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for label, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.mat'):
                    full_path = os.path.join(class_path, file)
                    mat = scipy.io.loadmat(full_path)

                    if self.key not in mat:
                        raise ValueError(f"Clé '{self.key}' introuvable dans le fichier {file}")

                    data = mat[self.key]  # shape (10000, 12)
                    if data.shape[0] < window_size:
                        continue

                    n_windows = data.shape[0] // window_size  # ex: 10000 // 200 = 50
                    features = []

                    for i in range(n_windows):
                        window = data[i * window_size : (i + 1) * window_size]  # (200, 12)

                        # ⚙️ Normalisation locale : max absolu par canal
                        max_per_channel = np.max(np.abs(window), axis=0) + 1e-8
                        normalized = window / max_per_channel  # shape (200, 12)

                        # 💡 Résumé de la fenêtre : moyenne par canal
                        feature_vector = np.mean(normalized, axis=0)  # shape (12,)
                        features.append(feature_vector)

                    sample = np.stack(features)  # shape (50, 12)
                    self.samples.append(sample)
                    self.labels.append(label)

        # ✅ Vérifie qu’il y a bien des données
        if len(self.samples) == 0:
            raise RuntimeError("Aucun fichier .mat valide trouvé.")

        # ✅ Calcul des stats globales pour normalisation (sur tous les (50,12))
        all_data = np.concatenate(self.samples, axis=0)  # shape (N*50, 12)
        self.global_mean = np.mean(all_data, axis=0)     # shape (12,)
        self.global_std = np.std(all_data, axis=0)       # shape (12,)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]  # shape (50, 12), dtype np.float32 normalement
        y = self.labels[idx]
    
        # ✅ Normalisation globale
        x = (x - self.global_mean) / (self.global_std + 1e-8)
    
        # ✅ Conversion en Tensors (float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)



# from torch.utils.data import DataLoader
# import torch

# # === Dataset personnalisé (défini auparavant) ===
# # from ton_module import DASMatDataset

# dataset = DASMatDataset(
#     root_dir=r"C:\Users\michel\Downloads\Das_data\train",
#     window_size=10000,
#     stride=10000,
#     key="data"
# )

# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # === Boucle sur tout le dataset ===
# total = 0
# for i, (x, y) in enumerate(train_loader):
#     print(f"Batch {i} → x: {x.shape}, y: {y.shape}")
#     total += x.shape[0]

# print(f"\nNombre total d'échantillons (fenêtres): {len(dataset)}")
# print(f"Nombre total vu dans DataLoader : {total}")
