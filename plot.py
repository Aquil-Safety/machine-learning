import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("artifacts/train_log.csv")

plt.plot(df["loss"], label="train_loss")
plt.plot(df["val_loss"], label="val_loss")
plt.legend()
plt.show()