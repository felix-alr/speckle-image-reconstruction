import random
from typing import Callable

import numpy as np
import scipy
import torch
import torch.nn.functional as func
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional


class OneHotTransform:
    def __init__(self, num_classes: int):
        """
        :param num_classes: The amount of classes of the classification problem.
        """
        self.num_classes = num_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of indices encoding one class per entry.
        :return: The one-hot-encoded tensor.
        """
        return torch.zeros(self.num_classes, dtype=float).scatter_(0, self.convert_to_tensor_if_int(x), 1.0)

    def convert_to_tensor_if_int(self, x) -> torch.Tensor:
        if isinstance(x, int):
            return torch.tensor([x])
        return x

class DiffuserTransform:
    def __init__(self, kernel: torch.Tensor):
        """
        :param kernel: The kernel used for convolution to perform the desired operation.
        """
        # Flip kernel for convolution
        self.kernel = kernel.flip(0, 1).view(1,1,*kernel.shape)

        # Calculate padding for convolution
        pad_h = self.kernel.shape[2]-1
        pad_w = self.kernel.shape[3]-1

        self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function computes a convolution of an input tensor x and the kernel passed on initialization.
        :param x: A tensor containing the data that shall be convoluted with self.kernel.
        :return: The computed convolution with self.kernel.
        """
        # Apply padding and perform convolution
        x = x.float()
        x = func.pad(x, self.padding)

        return func.conv2d(x, weight=self.kernel, padding=0, stride=1)


class RandomizedImageAugmentationTransform:
        def __call__(self, images: torch.Tensor, size_factor: int = 3, min_shift: int = -8, max_shift: int = 8, min_angle: int = 1, max_angle: int = 359) -> torch.Tensor:
            """
            :param images: The images that shall be augmented.
            :param size_factor: Proportionality constant defining the size of the augmented dataset depending on the
            size of images.
            :param min_shift: Minimum amount of pixels the image can be randomly shifted by.
            :param max_shift: Maximum amount of pixels the image can be randomly shifted by.
            :param min_angle: Minimum angle image can be randomly rotated by.
            :param max_angle: Maximum angle image can be randomly rotated by.
            :return: The augmented dataset containing the original images as well.
            """
            # Augmented set of images contains all the original images themselves
            augmented_set = [img for img in images]
            # Computing augmented images and appending them one by one
            for i in range((size_factor-1)*len(images)):
                augmented_set.append(self.compute_image(images[i%len(images)], min_shift, max_shift, min_angle, max_angle))
            # Return augmented dataset as torch.Tensor
            return torch.stack(augmented_set)

        def compute_image(self, image: torch.Tensor, min_shift: int, max_shift: int, min_angle: int, max_angle: int) -> torch.Tensor:
            """
            This function randomly shifts and rotates the input image according to the parameters.
            :param image: The image that shall be shifted and rotated.
            :param min_shift: Minimum amount of pixels the image can be randomly shifted by.
            :param max_shift: Maximum amount of pixels the image can be randomly shifted by.
            :param min_angle: Minimum angle image can be randomly rotated by.
            :param max_angle: Maximum angle image can be randomly rotated by.
            :return: A randomly shifted and rotated version of the input image.
            """
            # Roll the images by a random amount within specified bounds
            image = torch.roll(image, random.randint(a=min_shift, b=max_shift))
            # Rotate the images by a random angle within specified bounds
            image = functional.rotate(image, random.randint(a=min_angle,b=max_angle))
            return image

class ImageDatasetTargetAugmenter:
    def __init__(self, target_augmentation_transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x, feature_from_target_transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
        """
        :param target_augmentation_transform: The transform that multiplies and augments the target data as desired.
        :param feature_from_target_transform: The transform that constructs the feature from its corresponding label.
        """
        self.target_augmentation_transform = target_augmentation_transform
        self.feature_from_target_transform = feature_from_target_transform

    def __call__(self, targets: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        This function first augments the targets using self.target_augmentation_transform and the constructs the
        corresponding features using self.feature_from_target_transform.
        :param targets: The set of targets that  shall be augmented as determined by the transforms passed on
        initialization.
        :return: The augmented set of examples.
        """
        # Construct y_data from given targets using the augmentation transform
        y_data = self.target_augmentation_transform(targets)

        # Construct x_data from augmented targets using the target to x transform
        x_data = []
        for img in y_data:
            x_data.append(self.feature_from_target_transform(img))
        # Convert x_data back from numpy to torch.Tensor
        x_data = torch.from_numpy(np.array(x_data))
        return x_data, y_data

# Converts an array of images to a dataset where the training data is the target image from the given array passed through a diffuser transform
class SpeckleImageReconstructionDataset(Dataset):
    def __init__(self, images: torch.Tensor, diffuser_transform: DiffuserTransform, precompute: bool = True, target_shape: (int, int) = (16,16)):
        """
        :param images: The images that shall be used to construct the speckle image reconstruction dataset.
        :param diffuser_transform: The transform that will construct the feature tensor.
        :param precompute: A boolean indicating whether the data should be precomputed on initialization or computed
        on demand during execution.
        :param target_shape: The target shape of the images which they will be resized to.
        """
        self.images = images
        self.diffuser = diffuser_transform
        self.target_shape = target_shape
        self.precomputed_data = []

        # Precompute dataset if desired
        if precompute:
            print("Precomputing dataset...")
            for i in range(len(images)):
                self.precomputed_data.append(self.compute_item(i))
            print("Finished precomputing dataset!")

    def compute_item(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Computes one example for the dataset based on the index of an image from self.images.
        :param idx: The index of the image that shall be used to generate an example.
        :return: An example based on the desired image.
        """
        # Get image
        target = self.images[idx]
        # Normalize
        target = target.float() / 255.0
        # Modify to fit required shape
        target = target.unsqueeze(0)
        # Resize to 16x16
        resize_trans = transforms.Resize(self.target_shape)
        target = resize_trans(target)

        # Diffuse image if possible
        if self.diffuser is not None:
            input_img = self.diffuser(target).unsqueeze(0)
        else:
            input_img = target

        return input_img, target


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # Return from precomputed data if exists, otherwise compute on demand
        if self.precomputed_data:
            return self.precomputed_data[idx]
        else:
            return self.compute_item(idx)


class MatlabDataset(Dataset):
    # highest_value refers to the highest possible value of the data for normalization
    def __init__(self, mat_file: str, highest_value: float = 255.0, transform: Callable[[torch.Tensor],torch.Tensor] = lambda x: x, target_transform: Callable[[torch.Tensor],torch.Tensor] = lambda x: x, normalize: bool = True, train: bool = True):
        """
        Constructs a training or validation dataset as desired from a given .mat file.
        :param mat_file: The .mat file containing the dataset.
        :param highest_value: The theoretically highest value of the data entries for normalization.
        :param transform: A transform for the features.
        :param target_transform: A transform for the targets.
        :param normalize: A boolean indicating whether normalization is desired.
        :param train: A boolean indicating whether training or validation data shall be loaded.
        """
        mat_data = scipy.io.loadmat(mat_file)
        self.transform = transform
        self.target_transform = target_transform

        # Store desired data into x_data, y_data for further processing
        if train:
            x_data = mat_data['XTrain']
            y_data = mat_data['YTrain']
        else:
            x_data = mat_data['XValid']
            y_data = mat_data['YValid']

        # Reshape to convert from (SPATIAL, SPATIAL, CHANNEL, BATCH) to (BATCH, CHANNEL, SPATIAL, SPATIAL) for pytorch.
        self.x = self.transform(torch.from_numpy(x_data.transpose((3,2,0,1))).float())
        self.y = self.target_transform(torch.from_numpy(y_data.transpose((3,2,0,1))).float())

        # Skip further processing if normalization is not desired
        if not normalize:
            return

        # Normalize if desired
        if self.x.max() > 1.0:
            self.x /= highest_value
        if self.y.max() > 1.0:
            self.y /= highest_value

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]