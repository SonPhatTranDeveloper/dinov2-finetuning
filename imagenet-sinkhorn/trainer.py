import torch
import torch.optim as optim
import torch.nn as nn

import time


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, args):
        """
        Initialize the trainer for the DINOv2 ViT result
        """
        # Cache the parameters
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        # Model to device
        self.model.to(device)

        # Create optimizer and cross-entropy loss function
        self.optimizer = optim.Adam(self.model.parameters(), args["lr"])
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        """
        Train the Visual Transformer for one epoch
        :param epoch: the current epoch
        :return: epoch loss and accuracy
        """
        # Get the current time
        current_time = time.time()

        # Get the number of batches and the number of samples of the test loader
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        # Initialize the loss and accuracy
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        # Put the result into train mode
        self.model.train()

        # Calculate the loss and accuracy
        for image, label in self.train_loader:
            # Display debug message
            # print("Processing Image")

            # Map image and label to device
            image = image.to(self.device)
            label = label.to(self.device)

            # Forward pass through visual transformer
            output = self.model(image)
            loss = self.criterion(output, label)

            # Backward pass through visual transformer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate the loss and accuracy
            acc = (output.argmax(dim=1) == label).float().sum()
            epoch_accuracy += acc.item()
            epoch_loss += loss.item()

        # Calculate the loss and accuracy
        epoch_loss = epoch_loss / n_batches
        epoch_accuracy = epoch_accuracy / n_samples * 100

        # Calculate the training time
        print(time.time() - current_time)

        # Display the current status
        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, epoch_loss, epoch_accuracy))

        return epoch_accuracy, epoch_accuracy

    def validate(self, epoch):
        """
        Perform the validation at epoch
        :param epoch: the current epoch
        :return: the epoch loss and accuracy
        """
        # Get the number of batches and the number of samples of the test loader
        n_batches, n_samples = len(self.val_loader), len(self.val_loader.dataset)

        # Put the result into eval mode
        self.model.eval()

        # Validate
        with torch.no_grad():
            epoch_val_accuracy = 0.0
            epoch_val_loss = 0.0

            for data, label in self.val_loader:
                # Map image and label to device
                data = data.to(self.device)
                label = label.to(self.device)

                # Forward pass through the Visual Transformer
                val_output = self.model(data)
                val_loss = self.criterion(val_output, label)

                # Get the loss and accuracy
                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc.item()
                epoch_val_loss += val_loss.item()

        # Calculate the validation accuracy and loss
        epoch_val_loss = epoch_val_loss / n_batches
        epoch_val_accuracy = epoch_val_accuracy / n_samples * 100

        # Display the current stats
        print('Validation Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, epoch_val_loss, epoch_val_accuracy))

        return epoch_val_loss, epoch_val_accuracy

    def save(self, model_path, epoch):
        """
        Save the current result
        :param model_path: the saved result path
        :param epoch: the current epoch
        :return: None
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)