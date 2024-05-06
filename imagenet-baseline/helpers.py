# Define the helper class for fine-tuning with ImageNet dataset
from torchvision import transforms


class ResizeAndPad:
    def __init__(self, target_size, multiple):
        """
        Helper class to perform resize and padding on the image
        """
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        """
        Call transformation on the image
        """
        # Resize the image
        img = transforms.Resize(self.target_size)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = transforms.Pad(
            (pad_width // 2,
             pad_height // 2,
             pad_width - pad_width // 2,
             pad_height - pad_height // 2)
        )(img)

        return img