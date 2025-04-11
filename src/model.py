import torch
from torch import nn
from src.data_processing import prepare_data
from sklearn.model_selection import train_test_split

#DODAJ WAGI KLAS DLA ZROWNOWAZENIA MODELU ALE POTEM BO NIE WIEM
device = "cuda" if torch.cuda.is_available() else "cpu"

X, Y  = prepare_data()
print(X.shape)
print(Y.shape)

numpy_array_X, numpy_array_Y = X.values, Y.values
print(numpy_array_X)
print(numpy_array_Y)

X = torch.from_numpy(numpy_array_X).type(torch.float32)
Y = torch.from_numpy(numpy_array_Y).type(torch.int32)

print(X)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# .to(device) przenosi tensory (dane) na urządzenie,
# które jest określone w zmiennej device (czyli na GPU lub CPU).
# Przeniesienie danych na GPU pozwala na szybsze obliczenia,
# zwłaszcza w przypadku dużych danych.

x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

#sequential do tworzenia warstw

#ReLU dodaje nieliniowość do modelu, co pozwala na naukę bardziej złożonych funkcji.
#relu max(0, x)
#Oznacza to, że jeśli x jest większe od zera, to funkcja zwróci wartość
#x, w przeciwnym razie zwróci zero.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(in_features=10, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.neural_net(x)







