import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load result
    result = np.load("baseline.npy")

    # Extract training and testing
    train_loss_array, val_loss_array, train_accuracy_array, val_accuracy_array = (
        result
    )

    print(val_accuracy_array)

    # Draw the training graph
    epochs = list(range(50))
    plt.plot(epochs, train_accuracy_array, color="blue", linewidth=3, alpha=0.5, label="Train accuracy")
    plt.plot(epochs, val_accuracy_array, color="orange", linewidth=3, alpha=0.5, label="Validation accuracy")
    plt.title("Training and validation accuracy")

    # Display the plot
    plt.legend()
    plt.savefig("Sinkhorn Graph.png")
    plt.show()
