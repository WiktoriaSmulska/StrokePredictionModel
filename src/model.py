import torch
from torch import nn
from src.data_processing import prepare_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

X, Y = prepare_data()
X = torch.from_numpy(X.values).type(torch.float32)
Y = torch.from_numpy(Y.values).type(torch.float32)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# .to(device) przenosi tensory (dane) na urządzenie,
# które jest określone w zmiennej device (czyli na GPU lub CPU).
# Przeniesienie danych na GPU pozwala na szybsze obliczenia,
# zwłaszcza w przypadku dużych danych.

x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)


# - `sampling_strategy='minority'` → tylko **klasa mniejszościowa** (np. 1) będzie nadpróbkowana, aż zrówna się z klasą większościową.
# - `random_state=42` → żeby wyniki były powtarzalne (kontrola losowości).
# fit_resample(...) Tworzy nowe próbki klasy 1

smote = SMOTE(sampling_strategy='minority', random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train.cpu().numpy(), y_train.cpu().numpy())

x_train_res = torch.tensor(x_train_res, dtype=torch.float32).to(device)
y_train_res = torch.tensor(y_train_res, dtype=torch.float32).to(device)

print(f"Nowa liczba próbek klasy 1 w zbiorze treningowym: {y_train_res.sum().item()}")
print(f"Nowa liczba próbek klasy 0 w zbiorze treningowym: {(y_train_res == 0).sum().item()}")


#ReLU dodaje nieliniowość do modelu, co pozwala na naukę bardziej złożonych funkcji.
#relu max(0, x)
#Oznacza to, że jeśli x jest większe od zera, to funkcja zwróci wartość
#x, w przeciwnym razie zwróci zero.

#Warstwa 1: Uczy się prostych wzorców
#Warstwa 2: Łączy te cechy „osoba starsza + cukrzyca + nadciśnienie” → wysoki risk


#To tzw. "kaskadowe zwężanie" (funnel architecture) –
# sieć zaczyna od więcej neuronów,
# by uchwycić dużo cech, a potem je agreguje i upraszcza:


# Jeśli wejście do warstwy ma średnią 100 i odchylenie 50,
# BatchNorm je „spłaszczy” do czegoś bliżej średniej 0 i std 1.
# szybciej sie uczy przez to np

# Linear: oblicza wartości surowe (logity)
#
# BatchNorm: normalizuje je
#
# ReLU: nadaje nieliniowość

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neural_net(x)


model = Model().to(device)

epoches = 10000


#Jeśli masz np. 800 negatywnych przykładów i 200 pozytywnych,
# pos_weight będzie równe 800 / 200 = 4.
# Czyli błędy na klasie 1 będą 4 razy ważniejsze niż na klasie 0.

pos_weight_value = (y_train == 0).sum().item() / (y_train == 1).sum().item()
pos_weight_value *= 5
pos_weight = torch.tensor(pos_weight_value, device=device)


loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epoches):
    model.train()
    y_logits = model(x_train).squeeze()
    #surowa liczba z sieci neuronowej
    loss = loss_function(y_logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # with torch.no_grad():
    #     y_pred = torch.sigmoid(y_logits) >= 0.5
    #     correct = (y_pred == y_train).sum().item()
    #     accuracy = correct / y_train.size(0)
    #
    # print(f"Epoch {epoch + 1}/{epoches}")
    # print(f"Train Loss: {loss.item():.4f} | Train Accuracy: {accuracy * 100:.2f}%")
    # print("-" * 40)

    model.eval()
    with torch.inference_mode():
        y_logits_test = model(x_test)
        loss_test = loss_function(y_logits_test.squeeze(), y_test)

        y_test_pred = torch.sigmoid(y_logits_test.squeeze()) >= 0.5
        # sigmoid zmienia logity na wartości zprzedziału(0, 1) → interpretujemy to
        # jako prawdopodobieństwo klasy 1.

        correct_test = (y_test_pred == y_test).sum().item()
        test_accuracy = correct_test / y_test.size(0)

        print("\n" + "=" * 50)
        print("📊 TESTOWANIE MODELU NA DANYCH TESTOWYCH:")
        print(f"🧪 Test Loss: {loss_test.item():.4f}")
        print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"📈 Prawidłowe przewidywania: {correct_test} / {y_test.size(0)}")
        print("=" * 50 + "\n")


    y_test_cpu = y_test.cpu().numpy()
    y_pred_cpu = y_test_pred.cpu().numpy()

    print("\n📊 Szczegółowy raport klasyfikacji:")
    print(classification_report(y_test_cpu, y_pred_cpu, digits=4))

    print("🧮 Macierz pomyłek:")
    print(confusion_matrix(y_test_cpu, y_pred_cpu))









