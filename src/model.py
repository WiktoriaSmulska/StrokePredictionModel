import torch
from torch import nn
from src.data_processing import prepare_data
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

X, Y = prepare_data()

pd.set_option('display.max_columns', None)   # pokazuje wszystkie kolumny
pd.set_option('display.width', None)         # nie zawija do szerokości terminala
pd.set_option('display.max_colwidth', None)  # pełna zawartość komórek

X = torch.from_numpy(X.values).type(torch.float32)
Y = torch.from_numpy(Y.values).type(torch.float32)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neural_net(x)


model = Model().to(device)

epoches = 10000

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

accuracy = Accuracy(task="binary", num_classes=2)

for epoch in range(epoches):
    model.train()
    y_logits = model(x_train)

    loss = loss_function(y_logits.squeeze(), y_train)
    train_accuracy = accuracy(y_logits.squeeze(), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model.eval()
    with torch.inference_mode():
        y_logits_test = model(x_test)
        loss_test = loss_function(y_logits_test.squeeze(), y_test)
        test_accuracy = accuracy(y_logits_test.squeeze(), y_test)


        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} | Loss train: {loss} | Train accuracy {train_accuracy*100:.2f} Loss test: {loss_test} | Test accuracy {test_accuracy*100:.2f}")


    #     print("\n" + "=" * 50)
    #     print(" TESTOWANIE MODELU NA DANYCH TESTOWYCH:")
    #     print(f" Test Loss: {loss_test.item():.4f}")
    #     print(f" Test Accuracy: {test_accuracy * 100:.2f}%")
    #     print(f" Prawidłowe przewidywania: {correct_test} / {y_test.size(0)}")
    #     print("=" * 50 + "\n")
    #
    #
    # y_test_cpu = y_test.cpu().numpy()
    # y_pred_cpu = y_test_pred.cpu().numpy()
    #
    # print("\n Szczegółowy raport klasyfikacji:")
    # print(classification_report(y_test_cpu, y_pred_cpu, digits=4))
    #
    # print(" Macierz pomyłek:")
    # print(confusion_matrix(y_test_cpu, y_pred_cpu))









