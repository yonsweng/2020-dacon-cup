import pandas as pd
import numpy as np
import holidays
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1015)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


''' Load data '''
train = pd.read_csv('data/train.csv', encoding='euc-kr')
submission = pd.read_csv('data/submission.csv', encoding='euc-kr')


''' Get weights '''
w0 = train.iloc[:, 1].mean()
w1 = train.iloc[:, 2].mean()
w2 = train.iloc[:, 3].mean()
w3 = train.iloc[:, 4].mean()


''' Preprocess data '''
train['DateTime'] = pd.to_datetime(train.DateTime)
train['date'] = train['DateTime'].dt.date
train = train.groupby('date').sum().reset_index()
train['weekday'] = train['date'].apply(lambda x: x.weekday())
kr_holidays = holidays.KR()
train['holiday'] = train['date'].apply(lambda x: x in kr_holidays)
train.to_csv('train.csv', index=False, encoding='euc-kr')


''' Train-Valid split '''
valid = train.iloc[-31:, 1:]
train = train.iloc[:-31, 1:]


''' Prepare date '''
x = torch.tensor(train.iloc[:-1].values.reshape(-1, 1, 6).astype(np.float32),
                 dtype=torch.float, device=device)
y = torch.tensor(train.iloc[1:].values.reshape(-1, 1, 6)
                 .astype(np.float32), dtype=torch.float, device=device)
vx = np.concatenate([train.iloc[-1].values.reshape(-1, 1, 6),
                     np.zeros_like(valid.values.reshape(-1, 1, 6))])
vx = torch.tensor(vx.astype(np.float32), dtype=torch.float, device=device)
vy = torch.tensor(valid.values.reshape(-1, 1, 6)
                  .astype(np.float32), dtype=torch.float, device=device)
tx = np.concatenate([valid.iloc[-1:].values.reshape(-1, 1, 6),
                     np.zeros((submission.shape[0], 1, 6))])
tx = torch.tensor(tx.astype(np.float32), dtype=torch.float, device=device)


class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(6, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.rnn = nn.GRU(input_size=1024, hidden_size=1024, num_layers=2)
        self.l2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 4)
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, X, h_n=None):
        X = self.l1(X)
        X, h_n = self.rnn(X, h_n) if h_n is not None else self.rnn(X)
        X = self.dropout(X)
        X = self.l2(X)
        return X, h_n


def criterion(pred, true):
    # pred.shape = (N, 1, 4)
    # true.shape = (N, 1, 4)
    # w0, w1, w2, w3 <= train.csv의 사용자 수, 세션 수, 신규 방문자 수, 페이지 뷰 수 4가지 항목별 평균값
    loss = torch.sqrt(torch.mean(torch.square(
                true[:, :, 0] - pred[:, :, 0]))) / w0 + \
           torch.sqrt(torch.mean(torch.square(
                true[:, :, 1] - pred[:, :, 1]))) / w1 + \
           torch.sqrt(torch.mean(torch.square(
                true[:, :, 2] - pred[:, :, 2]))) / w2 + \
           torch.sqrt(torch.mean(torch.square(
                true[:, :, 3] - pred[:, :, 3]))) / w3
    return loss


def test(h_n):
    n = vx.shape[0] - 1
    for i in range(n):
        pred, h_n = model(vx[i:i+1], h_n)
        vx[i+1:i+2, :, :4] = pred
    loss = criterion(vx[1:, :, :4], vy)
    return loss


''' Training '''
num_epochs = 700
min_loss = 999.
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(1, num_epochs + 1):
    pred, h_n = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        v_loss = test(h_n.clone())
        print(f"Epochs {t}\t"
              f"Training loss: {loss.item():.3f}\t"
              f"Validation loss: {v_loss.item():.3f}", end='\t')
        if v_loss < min_loss:
            min_loss = v_loss
            torch.save(model.state_dict(), 'models/model.pt')
            saved_h_n = h_n
            print('Model saved', end='')
        print()


''' Prediction '''
# Pass validation data
model.load_state_dict(torch.load('models/model.pt'))
x = torch.cat([vx[:1], vy[:-1]], dim=0)
_, h_n = model(x, saved_h_n)

# Predict submission data
for i in range(submission.shape[0]):
    pred, h_n = model(tx[i:i+1], h_n)
    tx[i+1:i+2, :, :4] = pred

result = tx[1:, :, :4]
submission.iloc[:, 1:] = (result + 0.5).type(torch.long).reshape(-1, 4)\
    .cpu().numpy()
submission.to_csv('submission.csv', index=False, encoding='euc-kr')
