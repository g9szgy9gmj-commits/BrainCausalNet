import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import random
from sklearn import metrics

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)


NEG_LABEL = 0
POS_LABEL = 1
METRIC_NAMES = ["ACC", "AUC", "SEN", "SPEC", "F1"]


def calculate_binary_metrics(logits, labels):
    y_true = np.asarray(labels).astype(int).reshape(-1)
    y_score = F.softmax(logits, dim=1)[:, POS_LABEL].detach().cpu().numpy()
    y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL]).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = metrics.roc_auc_score(y_true, y_score)

    return {
        "ACC": metrics.accuracy_score(y_true, y_pred),
        "AUC": auc,
        "SEN": sen,
        "SPEC": spec,
        "F1": metrics.f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
    }


def average_metric_values(metric_values):
    averages = {}
    for name in METRIC_NAMES:
        values = np.asarray(metric_values[name], dtype=float)
        averages[name] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")
    return averages


def format_metric_values(metric_values):
    parts = []
    for name in METRIC_NAMES:
        value = metric_values[name]
        parts.append(f"{name}=nan" if np.isnan(value) else f"{name}={value:.4f}")
    return " ".join(parts)


class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        A = A.view(-1, self.in_channel, Nodes_number, Nodes_number)

        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)

        return concat1 + concat2


class Backbone(nn.Module):
    def __init__(self, dropout=0.5, in_channel=1):
        super().__init__()
        self.in_channel = in_channel

        self.e2e = nn.Sequential(
            E2E(self.in_channel, 8, (Nodes_number, Nodes_number)),
            nn.LeakyReLU(0.33),
            E2E(8, 8, (Nodes_number, Nodes_number)),
            nn.LeakyReLU(0.33),
        )

        self.e2n2g = nn.Sequential(
            nn.Conv2d(8, 48, (1, Nodes_number)),
            nn.LeakyReLU(0.33),
            nn.Conv2d(48, Nodes_number, (Nodes_number, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(Nodes_number, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
        )

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.e2e(x)
        x = self.e2n2g(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_A(self, x):
        x = self.e2e(x)
        x = torch.mean(x, dim=1)
        return x


class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1, in_channel=2):
        super().__init__()
        self.in_channel = in_channel

        # FC分支：先独立学特征
        self.fc_branch = Backbone(dropout=dropout, in_channel=1)

        # EC分支：先独立学特征
        self.ec_branch = Backbone(dropout=dropout, in_channel=1)

        # 单分支辅助任务头
        self.fc_head = nn.Linear(10, num_class)
        self.ec_head = nn.Linear(10, num_class)

        # 融合主任务头
        self.fusion_linear = nn.Sequential(
            nn.Linear(20, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )

        nn.init.kaiming_normal_(self.fc_head.weight)
        nn.init.zeros_(self.fc_head.bias)

        nn.init.kaiming_normal_(self.ec_head.weight)
        nn.init.zeros_(self.ec_head.bias)

        for layer in self.fusion_linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x shape: [B, 2, Nodes, Nodes]
        x_fc = x[:, 0:1, :, :]
        x_ec = x[:, 1:2, :, :]

        feat_fc = self.fc_branch(x_fc)
        feat_ec = self.ec_branch(x_ec)

        out_fc = self.fc_head(feat_fc)
        out_ec = self.ec_head(feat_ec)

        feat_fusion = torch.cat([feat_fc, feat_ec], dim=1)
        out_fusion = self.fusion_linear(feat_fusion)

        return out_fusion, out_fc, out_ec

    def get_A(self, x):
        x_fc = x[:, 0:1, :, :]
        x_ec = x[:, 1:2, :, :]

        A_fc = self.fc_branch.get_A(x_fc)
        A_ec = self.ec_branch.get_A(x_ec)

        return A_fc, A_ec


# In[]
# ABIDE load
print('loading ABIDE data...')
# X = np.load('./pcc_correlation_871_cc200.npy')
# Y = np.load('/kaggle/input/cc200-pytorch/871_label_cc200.npy')
# Data = scio.loadmat('./data/ABIDE/data/correlation/pcc_correlation_871_aal_.mat')
# print(cc200.keys()) # connectivity
# X = Data['connectivity']
# print(X[0][0])
# print(cc200.shape) # 871 200 200
# Y = np.loadtxt('./data/ABIDE/data/labels/871_labels.txt')

# X = np.load('./meta-paths/MDD-DDSC.npy')
# Y = np.load('./MDD_HC_label.npy')
X_fc = np.load('data/BD_HC/BD_HC.npy')
X_ec = np.load('output/BD_HC_EC_trimmed_0601_213204/20260607_201410/ec_all_subjects_trimmed_notopk.npy')
Y = np.load('data/BD_HC/BD_HC_label.npy')

assert X_fc.shape == X_ec.shape, f"Shape mismatch: FC {X_fc.shape} vs EC {X_ec.shape}"
X = np.stack([X_fc, X_ec], axis=1)  # (N, 2, Nodes, Nodes)
in_channel = X.shape[1]
Nodes_number = X.shape[-1]

# 保持原始含义：NaN->0, +Inf/-Inf->1
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=1.0)

print('---------------------')
print('X', X.shape)  # N M M
print('Y', Y.shape)
print('---------------------')

# In[]
epochs = 80  # 200 671
batch_size = 32  # 64 0.660
dropout = 0.5
lr = 0.005
decay = 0.01
result = []
result_final = []

from sklearn.model_selection import KFold

for ind in range(1):
    setup_seed(ind)
    fold_metric_values = {name: [] for name in METRIC_NAMES}
    kf = KFold(n_splits=10, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X, Y):
        kfold_index += 1
        print('kfold_index:', kfold_index)
        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]
        for train_index, val_index in kf.split(X_trainval, Y_trainval):
            X_train, X_val = X_trainval[:], X_trainval[:]
            Y_train, Y_val = Y_trainval[:], Y_trainval[:]
        print('X_train', X_train.shape)
        print('X_val', X_val.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_val', Y_val.shape)
        print('Y_test', Y_test.shape)

        # model
        model = Model(dropout=dropout, num_class=2, in_channel=in_channel)
        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        loss_fn = nn.CrossEntropyLoss()

        best_val = 0
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print("总参数数量和：" + str(k))

        # train
        for epoch in range(1, epochs + 1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            num_batch = X_train.shape[0] // int(batch_size)

            loss_train = 0
            for bn in range(num_batch):
                if bn == num_batch - 1:
                    batch = idx_batch[bn * int(batch_size):]
                else:
                    batch = idx_batch[bn * int(batch_size): (bn + 1) * int(batch_size)]

                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]
                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()

                out_fusion, out_fc, out_ec = model(train_data_batch_dev)

                loss_fusion = loss_fn(out_fusion, train_label_batch_dev)
                loss_fc = loss_fn(out_fc, train_label_batch_dev)
                loss_ec = loss_fn(out_ec, train_label_batch_dev)

                # 多任务总损失：先学特征，再融合；其余保持不变
                loss = loss_fusion + 0.2 * loss_fc + 0.3 * loss_ec

                loss_train += loss
                loss.backward()
                optimizer.step()

            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())

            # val
            if epoch % 1 == 0:
                if True:  # acc_val >= best_val:
                    model.eval()
                    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)

                    import time
                    start = time.time()

                    with torch.no_grad():
                        out_fusion, out_fc, out_ec = model(test_data_batch_dev)

                    end = time.time()

                    test_metrics = calculate_binary_metrics(out_fusion, Y_test)
                    print('Test', format_metric_values(test_metrics))

        result.append({"fold": kfold_index, **test_metrics})
        for name in METRIC_NAMES:
            fold_metric_values[name].append(test_metrics[name])

    result_final.append(average_metric_values(fold_metric_values))

final_metric_values = average_metric_values({
    name: [seed_result[name] for seed_result in result_final]
    for name in METRIC_NAMES
})
ACC = final_metric_values["ACC"]
AUC = final_metric_values["AUC"]
SEN = final_metric_values["SEN"]
SPEC = final_metric_values["SPEC"]
F1_SCORE = final_metric_values["F1"]

# In[]
print(result)
print(result_final)
print(format_metric_values(final_metric_values))
print("FINAL_ACC", ACC)
print("FINAL_AUC", AUC)
print("FINAL_SEN", SEN)
print("FINAL_SPEC", SPEC)
print("FINAL_F1", F1_SCORE)

# 0.656
