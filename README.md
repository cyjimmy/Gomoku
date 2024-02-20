## Gomoku
Gomoku is a two-player board game. The first player who places five stones in a row horizontally, vertically, or diagonally, wins the game.

This app trains an AI agent to play gomoku using convolutional neural network (CNN).

## Dataset
The gomoku dataset is acquired from Gomocup.org.

https://gomocup.org/results/gomocup-result-2022/

## Run gui app
1. Install required dependencis.
    ```
    pip install -r requirements.txt
    ```
2. Run `gui.py`
    ```
    python gui.py
    ```

## Train model (optional)
A trained model, `my_model.h5`, is included in the repo. The model is trained with 5000 games in `/dataset/raw`. If you want to train with more games to improve performance, you can follow steps below.
1. Run `create_dataset.py`. This will process the raw game files and compile them into a dataset file required to train the model. The output file will be stored in `/dataset/processed`.
    ```
    python create_dataset.py
    ```
2. Run `train_model.py`. This will train a CNN using the dataset created in previous step. The model will be stored in `/model`.
    ```
    python train_model.py
    ```
