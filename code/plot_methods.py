from matplotlib import pyplot as plt


def loss_plot(loss_list):
    plt.plot(loss_list)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.show()
