import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text

INPUT_PATH = "data/raw/synthetic_support_tickets_noisy.csv"
OUTPUT_PATH = "data/processed/clean_tickets.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    df["clean_message"] = df["customer_message"].apply(clean_text)

    # Remove empty messages
    df = df[df["clean_message"].str.strip().astype(bool)]

    df.to_csv(OUTPUT_PATH, index=False)
    print("Cleaned dataset saved:", df.shape)

if __name__ == "__main__":
    main()
