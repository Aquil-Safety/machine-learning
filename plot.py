"""Plot training and validation loss from artifacts/train_log.csv."""

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    """Render train/validation loss curves."""
    df = pd.read_csv("artifacts/train_log.csv")
    plt.plot(df["loss"], label="train_loss")
    plt.plot(df["val_loss"], label="val_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
