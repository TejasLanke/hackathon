from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[1]

CSV_PATH = Path(r"C:\Users\adity\Downloads\Dataset for Retinopathy\idrid_labels.csv")
IMAGE_DIR = Path(r"C:\Users\adity\Downloads\Dataset for Retinopathy\Imagenes\Imagenes")
OUTPUT_DIR = PROJECT / "data" / "raw"

df = pd.read_csv(CSV_PATH)

# Binary mapping: 0 = no DR, 1-4 = DR
df["class"] = df["diagnosis"].apply(lambda x: "no_dr" if x == 0 else "dr")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["class"]
)

for split, split_df in [("train", train_df), ("test", test_df)]:
    for _, row in split_df.iterrows():
        src = next(IMAGE_DIR.glob(f"{row['id_code']}.*"))

        dst_dir = OUTPUT_DIR / split / row["class"]
        dst_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst_dir / src.name)

print("Dataset preparation complete.")
print("Train:", len(train_df))
print("Test :", len(test_df))
print("\nTrain distribution:")
print(train_df["class"].value_counts())
print("\nTest distribution:")
print(test_df["class"].value_counts())