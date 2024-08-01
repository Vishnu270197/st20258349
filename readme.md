## Requirements

- Python 3.10 or later
- PyTorch
- Transformers
- Scikit-learn
- Matplotlib
- NumPy
- Pandas

## Installation


1. **Create a virtual environment and activate it**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset should be in CSV format with two columns: `word` and `tag`. Example:

| word  | tag     |
|-------|---------|
| EU    | B-ORG   |
| rejects| O      |
| German| B-MISC  |
| call  | O       |
| to    | O       |
| boycott| O      |
| British| B-MISC |
| lamb  | O       |
| .     | O       |

Ensure you have three CSV files: `train.csv`, `valid.csv`, and `test.csv` in a `datasets` directory.

## Usage

1. **Run the script**:

    ```bash
    python main.py
    ```

## Code Explanation

### Data Reading and Preprocessing

The function `read_csv_data(file_path)` reads the CSV files and converts them into a list of sentences where each sentence is a list of (word, tag) tuples.

### Tokenization and Dataset Preparation

The `preprocess_data(data, tokenizer, max_len=128)` function tokenizes the input data and prepares `TensorDataset` objects for training, validation, and testing.

### Model Training

The `train_model(model, train_loader, valid_loader, epochs=3)` function trains the BERT model using the training and validation data loaders. It also plots the training and validation loss over epochs.

### Model Evaluation

The `evaluate_model(model, test_loader)` function evaluates the trained model on the test data and prints the classification report.
