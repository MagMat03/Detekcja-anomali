import time
import pandas as pd
import numpy as np
import psutil
import os
import tracemalloc
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
  precision_score, recall_score, f1_score,
  confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Inicjalizacja monitoringu CPU
psutil.cpu_percent(interval=None)
start_total = time.time()
cpu_stats = []


def get_cpu_load():
  load = psutil.cpu_percent(interval=None)
  cpu_stats.append(load)
  return load


def get_cpu_temp():
  try:
      temps = psutil.sensors_temperatures()
      if temps:
          for key in ('cpu_thermal', 'coretemp', 'k10temp', 'acpitz'):
              if key in temps and temps[key]:
                  return temps[key][0].current
      # fallback: RPi
      path = "/sys/class/thermal/thermal_zone0/temp"
      if os.path.exists(path):
          with open(path) as f:
              return float(f.read().strip()) / 1000.0
  except Exception:
      pass
  return None



def get_power_watts():
  rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
  if not os.path.exists(rapl_path):
      return None, None
  try:
      def read_energy():
          with open(rapl_path) as f:
              return int(f.read().strip())


      e1 = read_energy();
      t1 = time.time()
      time.sleep(0.5)
      e2 = read_energy();
      t2 = time.time()
      idle_w = (e2 - e1) / 1e6 / (t2 - t1)
      return round(idle_w, 3), None
  except Exception:
      return None, None


# 1. Wczytanie i przygotowanie danych

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
DATA_PATH = r"C:\sensor.csv"

t0 = time.time()
print("Wczytywanie danych...")

df_raw = pd.read_csv(DATA_PATH)
X_raw = df_raw.drop(['Unnamed: 0', 'timestamp', 'machine_status'], axis=1, errors='ignore')
X_raw = X_raw.ffill().bfill().fillna(0)
y_raw = (df_raw['machine_status'] != 'NORMAL').astype(int)

if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
  print(f"Wczytywanie zapisanego podziału z '{TRAIN_FILE}' / '{TEST_FILE}'...")
  train_idx = np.loadtxt(TRAIN_FILE, dtype=int)
  test_idx = np.loadtxt(TEST_FILE, dtype=int)
else:
  print("Tworzenie nowego podziału i zapis do plików...")
  train_idx, test_idx = train_test_split(
      np.arange(len(X_raw)),
      test_size=0.3,
      shuffle=False  # zachowanie kolejności czasu
  )
  np.savetxt(TRAIN_FILE, train_idx, fmt='%d')
  np.savetxt(TEST_FILE, test_idx, fmt='%d')
  print(f"Zapisano: '{TRAIN_FILE}' ({len(train_idx)} próbek), '{TEST_FILE}' ({len(test_idx)} próbek)")


X_train = X_raw.iloc[train_idx]
X_test = X_raw.iloc[test_idx]
y_train = y_raw.iloc[train_idx]
y_test = y_raw.iloc[test_idx]


#  Sprawdzenie czy całe dane mieszczą się w pamięci
data_ram_mb = (X_raw.memory_usage(deep=True).sum() + y_raw.memory_usage(deep=True)) / 1024 ** 2
ram_total_mb = psutil.virtual_memory().total / 1024 ** 2
fits_in_ram = data_ram_mb < ram_total_mb * 0.8  # bezpieczny margines 80%


# Skalowanie
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]
hidden_dim = 8

get_cpu_load()
t_data = time.time() - t0
print(f"Czas przygotowania danych: {t_data:.2f} s")

# 2. Trening modelu (Autoencoder)

# Pomiar RAM modelu (przed i po)
tracemalloc.start()
t1 = time.time()
print("Trening modelu na zbiorze treningowym...")

# Pomiar mocy idle przed treningiem
idle_power_w, _ = get_power_watts()

model = MLPRegressor(
  hidden_layer_sizes=(32, 16, 32),
  activation='relu',
  solver='adam',
  max_iter=300,
  random_state=42
)
# Wybieramy tylko te wiersze, gdzie status to 0 (NORMAL)
mask_normal = (y_train == 0).values
X_train_normal_scaled = X_train_scaled[mask_normal]

# Trenujemy wyłącznie na poprawnych działaniach maszyny
model.fit(X_train_normal_scaled, X_train_normal_scaled)

t_train = time.time() - t1
_, model_ram_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
model_ram_mb = model_ram_peak / 1024 ** 2


get_cpu_load()
print(f"Czas treningu: {t_train:.2f} s")

# 3. Obliczanie progu i testowanie z latencją
t2 = time.time()
print("Testowanie modelu na danych niewidzianych...")

# Obliczamy MSE tylko dla danych, na których model się uczył
train_preds = model.predict(X_train_normal_scaled)
train_mse = np.mean(np.power(X_train_normal_scaled - train_preds, 2), axis=1)

# 98.8 percentyl zmniejszy ilość fałszywych alarmów (False Positives)
threshold = np.percentile(train_mse, 98.8)
# Pomiar latencji per-próbka
latencies_s = []
n_test = len(X_test_scaled)


# Pomiar mocy podczas detekcji (RAPL)
energy_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
rapl_available = os.path.exists(energy_path)
if rapl_available:
  with open(energy_path) as f:
      e_before_uj = int(f.read().strip())

t_detect_start = time.time()
test_preds = model.predict(X_test_scaled)
test_mse_all = np.mean(np.power(X_test_scaled - test_preds, 2), axis=1)

# Latencja per-próbka (mierzona osobno na losowych 200 próbkach, żeby uniknąć narzutu pętli)
sample_count = min(200, n_test)
sample_idx = np.random.choice(n_test, sample_count, replace=False)
for i in sample_idx:
  ts = time.perf_counter()
  _ = model.predict(X_test_scaled[i:i + 1])
  latencies_s.append(time.perf_counter() - ts)

t_detect_total = time.time() - t_detect_start

if rapl_available:
  with open(energy_path) as f:
      e_after_uj = int(f.read().strip())
  energy_total_j = (e_after_uj - e_before_uj) / 1e6
  load_power_w = energy_total_j / t_detect_total if t_detect_total > 0 else None
  energy_per_sample = energy_total_j / n_test if n_test > 0 else None
  energy_per_1000 = energy_per_sample * 1000 if energy_per_sample else None
else:
  load_power_w = None
  energy_per_sample = None
  energy_per_1000 = None

#Latencja – statystyki
lat_arr = np.array(latencies_s) * 1000  # → ms
lat_mean = np.mean(lat_arr)
lat_p95 = np.percentile(lat_arr, 95)
lat_p99 = np.percentile(lat_arr, 99)
throughput = n_test / t_detect_total  # próbki/s

# Klasyfikacja
pred_labels_test = (test_mse_all > threshold).astype(int)

precision = precision_score(y_test, pred_labels_test, zero_division=0)
recall = recall_score(y_test, pred_labels_test, zero_division=0)
f1 = f1_score(y_test, pred_labels_test, zero_division=0)

# RAM bufora danych testowych
buffer_ram_mb = (X_test_scaled.nbytes + test_mse_all.nbytes) / 1024 ** 2

get_cpu_load()
t_test = time.time() - t2
print(f"Czas testowania: {t_test:.2f} s")

# 4. Macierz pomyłek

print("Generowanie macierzy pomyłek...")
cm = confusion_matrix(y_test, pred_labels_test)
labels = ["NORMAL", "ANOMALIA"]
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title("Macierz pomyłek – zbiór testowy", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Zapisano: confusion_matrix.png")


# Czytelna tekstowa macierz
tn, fp, fn, tp = cm.ravel()
cm_text = f"""
Macierz pomyłek (Test Set):

                Pred: NORMAL   Pred: ANOMALIA
True: NORMAL       {tn:>8}        {fp:>8}
True: ANOMALIA     {fn:>8}        {tp:>8}

TN={tn}  FP={fp}  FN={fn}  TP={tp}
"""

# 5. Eksport wag do pliku .h (dla C++)
print("Eksport wag do model_weights.h...")


def to_c_array(arr, name):
    flat = arr.flatten()
    arr_str = ", ".join([f"{x:.6f}" for x in flat])
    return f"const float {name}[{len(flat)}] = {{{arr_str}}};\n"


with open("model_weights.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write(f"const int INPUT_DIM = {input_dim};\n")
    f.write(f"const int NUM_LAYERS = {len(model.coefs_)};\n")
    f.write(f"const float THRESHOLD = {threshold:.6f};\n\n")

    f.write(to_c_array(scaler.data_min_, "scaler_min"))
    f.write(to_c_array(scaler.data_range_, "scaler_range"))
    f.write("\n")

    # Automatyczna iteracja po wszystkich warstwach sieci (niezależnie od ich liczby)
    for i in range(len(model.coefs_)):
        f.write(f"// Warstwa {i + 1}\n")
        f.write(to_c_array(model.coefs_[i], f"WEIGHTS_{i + 1}"))
        f.write(to_c_array(model.intercepts_[i], f"BIAS_{i + 1}"))
        f.write("\n")

    f.write("#endif\n")


# 6. Raport końcowy
end_total = time.time()
avg_cpu = sum(cpu_stats) / len(cpu_stats) if cpu_stats else 0
max_cpu = max(cpu_stats) if cpu_stats else 0
ram_info = psutil.virtual_memory()
ram_used_p = ram_info.percent
ram_used_mb = (ram_info.total - ram_info.available) / 1024 ** 2

cpu_temp = get_cpu_temp()
temp_str = f"{cpu_temp:.1f} °C" if cpu_temp is not None else "niedostępna"

# Throttling (RPi: /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
throttling = "nie wykryto"
try:
  cur_freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
  max_freq_path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
  if os.path.exists(cur_freq_path) and os.path.exists(max_freq_path):
      with open(cur_freq_path) as f: cur_f = int(f.read())
      with open(max_freq_path) as f: max_f = int(f.read())
      ratio = cur_f / max_f
      throttling = f"TAK (obecna: {cur_f // 1000} MHz / max: {max_f // 1000} MHz, {ratio * 100:.1f}%)" \
          if ratio < 0.95 else \
          f"NIE ({cur_f // 1000} MHz / {max_f // 1000} MHz, {ratio * 100:.1f}%)"
except Exception:
  pass


def fmt_power(val):
  return f"{val:.3f} W" if val is not None else "niedostępny (brak RAPL)"

def fmt_energy(val):
  return f"{val:.6f} J" if val is not None else "niedostępna (brak RAPL)"

raport = f"""
╔══════════════════════════════════════════════════════════════════╗
║        RAPORT – MLP AUTOENCODER – DETEKCJA ANOMALII             ║
║        Data: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║
╚══════════════════════════════════════════════════════════════════╝




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PODZIAŁ DANYCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Próbki treningowe:          {len(X_train_scaled):>10}
Próbki testowe:             {len(X_test_scaled):>10}
Liczba sensorów (cech):     {input_dim:>10}
Pliki podziału:             train.txt / test.txt




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRYKI KLASYFIKACJI (zbiór testowy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Próg anomalii (Threshold):  {threshold:.6f}
Średnie MSE (Test):         {np.mean(test_mse_all):.6f}
Precision (Precyzja):       {precision:.4f}
Recall (Czułość):           {recall:.4f}
F1-score:                   {f1:.4f}
{cm_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CZAS PRACY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Przygotowanie danych:       {t_data:.4f} s
Uczenie modelu:             {t_train:.4f} s
Detekcja (cały zbiór test): {t_detect_total:.4f} s
Czas detekcji 1 próbki:     {t_detect_total / n_test * 1000:.4f} ms




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LATENCJA (pomiar na {sample_count} losowych próbkach)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Średnia latencja:           {lat_mean:.4f} ms
P95 latencja:               {lat_p95:.4f} ms
P99 latencja:               {lat_p99:.4f} ms
Przepustowość:              {throughput:.2f} próbek/s




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAMIĘĆ RAM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zużycie RAM (system):       {ram_used_mb:.1f} MB  ({ram_used_p:.1f}%)
RAM zużyty przez model:     {model_ram_mb:.3f} MB
RAM bufora danych test:     {buffer_ram_mb:.3f} MB
Rozmiar całego datasetu:    {data_ram_mb:.2f} MB
Czy mieści się w pamięci:   {"TAK" if fits_in_ram else "NIE – ryzyko swapowania!"}




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Średnie użycie CPU:         {avg_cpu:.1f}%
Maksymalne użycie CPU:      {max_cpu:.1f}%
Temperatura CPU:            {temp_str}
Throttling:                 {throttling}




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POBÓR MOCY I ENERGIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pobór mocy (idle):          {fmt_power(idle_power_w)}
Pobór mocy (detekcja):      {fmt_power(load_power_w)}
Energia na 1 próbkę:        {fmt_energy(energy_per_sample)}
Energia na 1000 próbek:     {fmt_energy(energy_per_1000)}




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLIKI WYJŚCIOWE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pelny_raport.txt            – ten raport
model_weights.h             – wagi dla C++
confusion_matrix.png        – macierz pomyłek
train.txt / test.txt        – indeksy podziału
"""

print(raport)

with open("pelny_raport.txt", "w", encoding="utf-8") as f:
  f.write(raport)

print(">>> GOTOWE! Pliki zapisane.")



