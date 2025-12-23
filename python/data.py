import scipy
import torch
import torch.nn.functional as func
from torch.utils.data import Dataset
from torchvision import transforms


class OneHotTransform:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        return torch.zeros(self.num_classes, dtype=float).scatter_(0, self.convert_to_tensor_if_int(x), 1.0)

    def convert_to_tensor_if_int(self, x):
        if isinstance(x, int):
            return torch.tensor([x])
        return x

class DiffuserTransform:
    def __init__(self, kernel):
        # Flip kernel for convolution
        self.kernel = kernel.flip(0, 1).view(1,1,*kernel.shape)

        # Calculate padding for convolution
        pad_h = self.kernel.shape[2]-1
        pad_w = self.kernel.shape[3]-1

        self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    def __call__(self, x):
        # Apply padding and perform convolution
        x = x.float()
        x = func.pad(x, self.padding)

        return func.conv2d(x, weight=self.kernel, padding=0, stride=1).squeeze(0)


# Converts an array of images to a dataset where the training data is the target image from the given array passed through a diffuser transform
class SpeckleImageReconstructionDataset(Dataset):
    def __init__(self, images, diffuser_transform, precompute = True, target_shape = (16,16)):
        self.images = images
        self.diffuser = diffuser_transform
        self.target_shape = target_shape
        self.precomputed_data = []

        if precompute:
            print("Precomputing dataset...")
            for i in range(len(images)):
                self.precomputed_data.append(self.compute_item(i))
            print("Finished precomputing dataset...")

    def compute_item(self, idx):
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


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.precomputed_data:
            return self.precomputed_data[idx]
        else:
            return self.compute_item(idx)


class MatlabDataset(Dataset):
    # highest_value refers to the highest possible value of the data for normalization
    def __init__(self, mat_file, highest_value = 255.0, normalize = True, train = True):
        mat_data = scipy.io.loadmat(mat_file)

        if train:
            x_data = mat_data['XTrain']
            y_data = mat_data['YTrain']
        else:
            x_data = mat_data['XValid']
            y_data = mat_data['YValid']

        self.x = torch.from_numpy(x_data.transpose((3,2,0,1))).float()
        self.y = torch.from_numpy(y_data.transpose((3,2,0,1))).float()

        if not normalize:
            return

        if self.x.max() > 1.0:
            self.x /= highest_value
        if self.y.max() > 1.0:
            self.y /= highest_value

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]