import pandas as pd
import re

# Load raw data
df = pd.read_csv("data/raw/news_data.csv")

def clean_text(text):
    text = text.lower()                     # lowercase
    text = re.sub(r"[^a-z\s]", "", text)    # remove symbols
    text = re.sub(r"\s+", " ", text)        # remove extra spaces
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

df.to_csv("data/cleaned/news_cleaned.csv", index=False)

print("Text cleaning completed!")
