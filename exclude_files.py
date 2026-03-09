import os
import pandas as pd

# Pobranie ścieżki bazowej od użytkownika
base_path = input("Podaj ścieżkę bazową (tam gdzie jest train.csv i train_images): ").strip()

csv_path = os.path.join(base_path, "train.csv")
images_dir = os.path.join(base_path, "train_images")

# Wczytanie pliku CSV
df = pd.read_csv(csv_path)

# Iteracja po wierszach
for _, row in df.iterrows():
    id_code = row["id_code"]
    diagnosis = row["diagnosis"]

    # Usuwamy jeśli diagnosis nie jest 0 ani 4
    if diagnosis not in [0, 4]:
        img_path = os.path.join(images_dir, f"{id_code}.png")

        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Usunięto: {img_path}")
        else:
            print(f"Brak pliku: {img_path}")
