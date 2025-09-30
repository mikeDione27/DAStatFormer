from torch.utils.data import Dataset
import torch
from scipy.io import loadmat

class MyDataset(Dataset):
    def __init__(self, path: str, dataset: str):
        super(MyDataset, self).__init__()
        self.dataset = dataset
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.test_dataset, \
        self.test_label, \
        self.max_length_sample_inTest, \
        self.train_dataset_with_no_paddding = self.pre_option(path)

    def __getitem__(self, index):
        if self.dataset == 'train':
            x = self.train_dataset[index]
            y = self.train_label[index].long()
        elif self.dataset == 'test':
            x = self.test_dataset[index]
            y = self.test_label[index].long()
        return x, y

    def __len__(self):
        return self.train_len if self.dataset == 'train' else self.test_len

    def pre_option(self, path: str):
        m = loadmat(path)
        data = m[list(m.keys())[-1]]  # last key contains data
        data00 = data[0][0]

        # Automatically find correct indices
        keys = [key[0] for key in data.dtype.descr]
        index_train = keys.index('train')
        index_trainlabels = keys.index('trainlabels')
        index_test = keys.index('test')
        index_testlabels = keys.index('testlabels')

        train_data = data00[index_train].squeeze()
        train_label = data00[index_trainlabels].squeeze()
        test_data = data00[index_test].squeeze()
        test_label = data00[index_testlabels].squeeze()

        train_len = train_data.shape[0]
        test_len = test_data.shape[0]
        output_len = len(set(train_label.tolist()))

        # max time steps
        max_len = max(max(x.shape[1] for x in train_data), max(x.shape[1] for x in test_data))

        train_dataset = []
        test_dataset = []
        train_dataset_no_padding = []
        test_dataset_no_padding = []
        max_length_sample_inTest = []

        for x in train_data:
            train_dataset_no_padding.append(x.transpose(-1, -2).tolist())
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[1] != max_len:
                padding = torch.zeros(x.shape[0], max_len - x.shape[1])
                x = torch.cat((x, padding), dim=1)
            train_dataset.append(x)

        for x in test_data:
            test_dataset_no_padding.append(x.transpose(-1, -2).tolist())
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[1] != max_len:
                padding = torch.zeros(x.shape[0], max_len - x.shape[1])
                x = torch.cat((x, padding), dim=1)
            else:
                max_length_sample_inTest.append(x.transpose(-1, -2))
            test_dataset.append(x)

        train_dataset = torch.stack(train_dataset).permute(0, 2, 1)  # (N, T, C)
        test_dataset = torch.stack(test_dataset).permute(0, 2, 1)
        train_label = torch.tensor(train_label, dtype=torch.long)
        test_label = torch.tensor(test_label, dtype=torch.long)

        channel = test_dataset.shape[-1]
        input_len = test_dataset.shape[-2]

        return train_len, test_len, input_len, channel, output_len, \
               train_dataset, train_label, test_dataset, test_label, \
               max_length_sample_inTest, train_dataset_no_padding
               
               


import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class MyDataset_3d(Dataset):
    def __init__(self, path: str, dataset: str):
        super(MyDataset_3d, self).__init__()
        self.dataset = dataset
        self.temporal_idx = list(range(0, 11))     # 11 time-domain features
        self.waveform_idx = list(range(11, 19))    # 8 waveform-domain features
        self.spectral_idx = list(range(19, 24))    # 5 spectral-domain features

        (
            self.train_len,
            self.test_len,
            self.input_len,
            self.channel_len,
            self.output_len,
            self.train_dataset,   # tensors (N_train, T_max, C) NON normalisés
            self.train_label,
            self.test_dataset,    # tensors (N_test,  T_max, C) NON normalisés
            self.test_label
        ) = self._load_mat(path)

    def __getitem__(self, index):
        # --- Normalisation à la volée, juste avant l'entraînement ---
        if self.dataset == 'train':
            x = self.train_dataset[index].clone()     # (T, C)
            y = self.train_label[index].long()
            L = self.train_lengths[index]             # longueur valide
        else:
            x = self.test_dataset[index].clone()
            y = self.test_label[index].long()
            L = self.test_lengths[index]

        # Normaliser uniquement la partie valide (0..L-1) ; garder le padding à 0
        # mean/std sont (C,), broadcasting sur l'axe C
        x[:L] = (x[:L] - self.norm_mean_) / self.norm_std_
        x[L:] = 0.0

        return x, y

    def __len__(self):
        return self.train_len if self.dataset == 'train' else self.test_len

    # ----------------- Helpers internes -----------------
    def _unwrap_label_array(self, arr):
        arr = np.asarray(arr).squeeze()
        if arr.dtype == object:
            arr = np.array([int(np.array(v).squeeze()) for v in arr.ravel()], dtype=np.int64)
        else:
            arr = arr.astype(np.int64, copy=False).ravel()
        return arr

    def _as_CT(self, x, C_expected=None):
        """Assure le format (C,T). Transpose si (T,C_expected)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Chaque sample doit être 2D, reçu shape={x.shape}")
        if C_expected is None:
            return x
        if x.shape[0] == C_expected:
            return x
        if x.shape[1] == C_expected:
            return x.T
        return x  # sinon, on laisse tel quel

    def _compute_channel_stats(self, train_cells_CT, C):
        """μ/σ par canal sur TRAIN uniquement, sans padding."""
        sum_c   = np.zeros(C, dtype=np.float64)
        sumsq_c = np.zeros(C, dtype=np.float64)
        count_c = np.zeros(C, dtype=np.int64)
        for x in train_cells_CT:
            sum_c   += x.sum(axis=1)
            sumsq_c += (x * x).sum(axis=1)
            count_c += x.shape[1]
        eps = 1e-8
        mean = sum_c / np.maximum(count_c, 1)
        var  = np.maximum(sumsq_c / np.maximum(count_c, 1) - mean**2, eps)
        std  = np.sqrt(var)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def _pad_and_stack_TxC(self, list_CT):
        """Pad à droite sur T, renvoie tensor (N, T_max, C)."""
        C = list_CT[0].shape[0]
        T_max = max(x.shape[1] for x in list_CT)
        stacked = []
        for x in list_CT:
            if x.shape[1] < T_max:
                pad = np.zeros((C, T_max - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            stacked.append(torch.from_numpy(x).permute(1, 0))  # (T, C)
        return torch.stack(stacked)  # (N, T_max, C)

    # ----------------- Chargement du .mat (sans normaliser) -----------------
    def _load_mat(self, path):
        m = loadmat(path)
        data = m[list(m.keys())[-1]]  # struct principal (ex: 'GTN_dataset')
        data00 = data[0][0]
        keys = [k[0] for k in data.dtype.descr]

        train_cells = data00[keys.index('train')].squeeze()       # (N_train,) objets
        test_cells  = data00[keys.index('test')].squeeze()        # (N_test,)  objets
        y_train_raw = data00[keys.index('trainlabels')].squeeze()
        y_test_raw  = data00[keys.index('testlabels')].squeeze()

        # Labels propres (N,)
        y_train = self._unwrap_label_array(y_train_raw)
        y_test  = self._unwrap_label_array(y_test_raw)
        output_len = int(np.unique(y_train).size)

        # Déduire C et forcer (C,T) pour chaque sample (sans padding, sans normalisation)
        C = int(np.asarray(train_cells[0]).shape[0])
        train_CT = [self._as_CT(x, C) for x in train_cells]
        test_CT  = [self._as_CT(x, C) for x in test_cells]

        # Longueurs réelles (pour normaliser uniquement la partie valide)
        self.train_lengths = [x.shape[1] for x in train_CT]
        self.test_lengths  = [x.shape[1] for x in test_CT]

        # μ/σ calculés sur TRAIN uniquement (sans padding) — stockés pour __getitem__
        mean, std = self._compute_channel_stats(train_CT, C)
        self.norm_mean_ = torch.from_numpy(mean)  # (C,)
        self.norm_std_  = torch.from_numpy(std)   # (C,)

        # Padding puis empilement en (N, T_max, C) — NON normalisé
        train_tensor = self._pad_and_stack_TxC(train_CT)
        test_tensor  = self._pad_and_stack_TxC(test_CT)

        input_len = train_tensor.shape[1]   # T_max
        channel_len = train_tensor.shape[2] # C
        train_len = train_tensor.shape[0]
        test_len  = test_tensor.shape[0]

        # Tensors labels
        train_label = torch.from_numpy(y_train).long()
        test_label  = torch.from_numpy(y_test).long()

        return (
            train_len, test_len, input_len, channel_len, output_len,
            train_tensor, train_label, test_tensor, test_label
        )


from torch.utils.data import Dataset
import torch
import numpy as np

class MyDataset_npz(Dataset):
    def __init__(self, path: str, dataset: str):
        super(MyDataset_npz, self).__init__()
        self.dataset = dataset
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.train_label, \
        self.test_dataset, \
        self.test_label, \
        self.max_length_sample_inTest, \
        self.train_dataset_with_no_padding = self.pre_option(path)

    def __getitem__(self, index):
        if self.dataset == 'train':
            x = self.train_dataset[index]
            y = self.train_label[index].long()
        elif self.dataset == 'test':
            x = self.test_dataset[index]
            y = self.test_label[index].long()
        return x, y

    def __len__(self):
        return self.train_len if self.dataset == 'train' else self.test_len

    def pre_option(self, path: str):
        data = np.load(path, allow_pickle=True)

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        train_len = len(X_train)
        test_len = len(X_test)
        output_len = len(np.unique(y_train))

        # Trouver la longueur maximale en temps (dimension 1)
        max_len = max(
            max(x.shape[1] for x in X_train),
            max(x.shape[1] for x in X_test)
        )

        train_dataset = []
        test_dataset = []
        train_dataset_no_padding = []
        test_dataset_no_padding = []
        max_length_sample_inTest = []

        for x in X_train:
            
            train_dataset_no_padding.append(x.transpose(1, 0).tolist())
            
            x = np.asarray(x, dtype=np.float16)
            x = torch.tensor(x, dtype=torch.float16)


            if x.shape[1] != max_len:
                padding = torch.zeros(x.shape[0], max_len - x.shape[1])
                x = torch.cat((x, padding), dim=1)
            train_dataset.append(x)

        for x in X_test:
            
            test_dataset_no_padding.append(x.transpose(1, 0).tolist())
           
            x = np.asarray(x, dtype=np.float16)
            x = torch.tensor(x, dtype=torch.float16)

            if x.shape[1] != max_len:
                padding = torch.zeros(x.shape[0], max_len - x.shape[1])
                x = torch.cat((x, padding), dim=1)
            else:
                max_length_sample_inTest.append(x.transpose(1, 0))
            test_dataset.append(x)

        train_dataset = torch.stack(train_dataset).permute(0, 2, 1)  # (N, T, C)
        test_dataset = torch.stack(test_dataset).permute(0, 2, 1)
        train_label = torch.tensor(y_train, dtype=torch.long)
        test_label = torch.tensor(y_test, dtype=torch.long)

        channel = test_dataset.shape[-1]
        input_len = test_dataset.shape[-2]

        return train_len, test_len, input_len, channel, output_len, \
               train_dataset, train_label, test_dataset, test_label, \
               max_length_sample_inTest, train_dataset_no_padding
