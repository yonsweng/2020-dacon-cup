import pandas as pd
import numpy as np
import holidays
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(1015)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


''' Load data '''
train = pd.read_csv('data/train.csv', encoding='euc-kr')
new_train = pd.read_csv('data/new_train.csv', encoding='euc-kr')
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
# train.to_csv('train.csv', index=False, encoding='euc-kr')

new_train['DateTime'] = pd.to_datetime(new_train.DateTime)
new_train['date'] = new_train['DateTime'].dt.date
new_train = new_train.groupby('date').sum().reset_index()
new_train['weekday'] = new_train['date'].apply(lambda x: x.weekday())
new_train['holiday'] = new_train['date'].apply(lambda x: x in kr_holidays)

submission['DateTime'] = pd.to_datetime(submission.DateTime)
submission['DateTime'] = submission['DateTime'].dt.date
submission['weekday'] = submission['DateTime'].apply(lambda x: x.weekday())
submission['holiday'] = \
    submission['DateTime'].apply(lambda x: x in kr_holidays)

''' Train-Valid split '''
valid = new_train.iloc[:, 1:]
train = train.iloc[:, 1:]


''' Normalization '''
scaler = StandardScaler()
train.iloc[:, :4] = scaler.fit_transform(train.iloc[:, :4])
valid.iloc[:, :4] = scaler.transform(valid.iloc[:, :4])
min_max_scaler = MinMaxScaler()
train.iloc[:, 4:5] = min_max_scaler.fit_transform(train.iloc[:, 4:5])
valid.iloc[:, 4:5] = min_max_scaler.transform(valid.iloc[:, 4:5])
submission['weekday'] = min_max_scaler.transform(
    submission['weekday'].values.reshape(-1, 1)).reshape(-1)


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
                     np.zeros((31, 1, 6))])
tx[-31:, :, 4] = submission.iloc[-31:]['weekday'].values.reshape(-1, 1)
tx[-31:, :, 5] = submission.iloc[-31:]['holiday'].values.reshape(-1, 1)
tx = torch.tensor(tx.astype(np.float32), dtype=torch.float, device=device)


class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.rnn = nn.GRU(input_size=512, hidden_size=512, num_layers=2)
        self.l2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
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
    model.eval()
    n = vx.shape[0] - 1
    for i in range(n):
        pred, h_n = model(vx[i:i+1], h_n)
        vx[i+1, :, :4] = pred.reshape(1, 4)
    loss = criterion(vx[1:, :, :4], vy)
    return loss


''' Training '''
patience = 10
n_reduce = 5
counter = 0
num_epochs = 500
min_loss = 999.
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
writer = SummaryWriter()

for t in range(1, num_epochs + 1):
    pred, h_n = model(x)
    loss = criterion(pred, y[:, :, :4])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        v_loss = test(h_n.clone())
        print(f"Epochs {t}\t"
              f"Training loss: {loss.item():.3f}\t"
              f"Validation loss: {v_loss.item():.3f}", end='\t')
        writer.add_scalar('t_loss', loss.item(), t)
        writer.add_scalar('v_loss', v_loss, t)

        if v_loss < min_loss:
            min_loss = v_loss
            torch.save(model.state_dict(), 'models/model.pt')
            saved_h_n = h_n
            print('Model saved', end='')
            counter = 0
        else:
            if counter == patience:
                counter = 0
                n_reduce -= 1
                if n_reduce == 0:
                    print()
                    break
                else:
                    # model.load_state_dict(torch.load('models/model.pt'))
                    print('lr reduced', end='')
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.25
            counter += 1
        print()
        model.train()


''' Prediction '''
# Pass validation data
model.load_state_dict(torch.load('models/model.pt'))
model.eval()
x = torch.cat([vx[:1], vy[:-1]], dim=0)
_, h_n = model(x, saved_h_n)

# Predict submission data
for i in range(31):
    pred, h_n = model(tx[i:i+1], h_n)
    tx[i+1, :, :4] = pred.reshape(1, 4)

result = tx[1:, :, :4].reshape(-1, 4).detach().cpu().numpy()
result = scaler.inverse_transform(result)
submission.iloc[:-31, 1:5] = \
    scaler.inverse_transform(valid.iloc[:, :4]).astype(int)
submission.iloc[-31:, 1:5] = (result + 0.5).astype(int)
submission = submission.iloc[:, 0:5]
submission.to_csv('submission.csv', index=False, encoding='euc-kr')

writer.close()
