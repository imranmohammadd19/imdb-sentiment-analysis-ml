import pandas as pd
import sqlite3

# Load IMDb CSV
df = pd.read_csv("data/imdb_raw.csv")

# Convert sentiment to numeric
df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Connect to SQLite database (creates file automatically)
conn = sqlite3.connect("database/imdb.db")

# Store data into table
df.to_sql("reviews", conn, if_exists="replace", index=False)

conn.close()

print("Data successfully loaded into SQLite!")

