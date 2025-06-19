# 画曲线
from matplotlib import pyplot as plt


def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = "Epoch"
    legends = ["Training", "Validation"]

    plt.figure(figsize=(20, 5))

    y1 = history.history["accuracy"]
    y2 = history.history["val_accuracy"]

    min_y = min(min(y1), min(y2)) - ylim_pad[0]
    max_y = max(max(y1), max(y2)) + ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title("Model Accuracy\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc="upper left")
    plt.grid()

    y1 = history.history["loss"]
    y2 = history.history["val_loss"]

    min_y = min(min(y1), min(y2)) - ylim_pad[1]
    max_y = max(max(y1), max(y2)) + ylim_pad[1]

    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title("Model Loss:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc="upper left")
    plt.grid()
    plt.show()
