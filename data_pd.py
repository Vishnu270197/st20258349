import pandas as pd

# Sample data for train, validation, and test sets
train_data = [
    ("EU", "B-ORG"),
    ("rejects", "O"),
    ("German", "B-MISC"),
    ("call", "O"),
    ("to", "O"),
    ("boycott", "O"),
    ("British", "B-MISC"),
    ("lamb", "O"),
    (".", "O"),
    (None, None),
    ("Peter", "B-PER"),
    ("Blackburn", "I-PER"),
]

valid_data = [
    ("France", "B-LOC"),
    ("is", "O"),
    ("a", "O"),
    ("country", "O"),
    ("in", "O"),
    ("Europe", "B-LOC"),
    (".", "O"),
    (None, None),
    ("John", "B-PER"),
    ("Doe", "I-PER"),
]

test_data = [
    ("Microsoft", "B-ORG"),
    ("announced", "O"),
    ("a", "O"),
    ("new", "O"),
    ("product", "O"),
    (".", "O"),
    (None, None),
    ("Alice", "B-PER"),
    ("Smith", "I-PER"),
]

# Convert to DataFrames
train_df = pd.DataFrame(train_data, columns=["word", "tag"])
valid_df = pd.DataFrame(valid_data, columns=["word", "tag"])
test_df = pd.DataFrame(test_data, columns=["word", "tag"])

# Save to CSV files
train_df.to_csv("train.csv", index=False)
valid_df.to_csv("valid.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("CSV files created.")
