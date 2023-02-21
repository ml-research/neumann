import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def load_images_and_labels(dataset='member', split='train', base=None):
    """Load image paths and labels for clevr-hans dataset.
    """
    image_paths = []
    label_paths = []
    base_folder = 'data/vqa/' + dataset + '/' + split + '/'
    folder_names = sorted(os.listdir(base_folder))
    if '.DS_Store' in folder_names:
        folder_names.remove('.DS_Store')

    for folder_name in folder_names:
        folder = base_folder + folder_name + '/'
        image_paths.append(os.path.join(folder, 'scene.png'))
        label_paths.append(os.path.join(folder, 'answer.txt'))
    return image_paths, label_paths


def load_image_clevr(path):
    """Load an image using given path.
    """
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img

def load_answer(path):
    """Load answers from the text file.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    line = lines[0]
    # question red
    # answer = line.split(',')[0]
    # question gray
    # answer = line.split(',')[2]
    # question yellow
    answer = line.split(',')[-1]
    if answer == 'red':
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    elif answer == 'gray':
        return torch.tensor([0.0, 1.0, 0.0, 0.0])
    elif answer == 'cyan':
        return torch.tensor([0.0, 0.0, 1.0, 0.0])
    elif answer == 'yellow':
        return torch.tensor([0.0, 0.0, 0.0, 1.0])
    else:
        assert 0, "Invalid answer in {}: {}".format(path, answer)



class VQA(torch.utils.data.Dataset):
    def __init__(self, dataset, split, img_size=128, base=None):
        super().__init__()
        self.img_size = img_size
        self.dataset = dataset
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.split = split
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size))]
        )
        self.image_paths, self.answer_paths = load_images_and_labels(
            dataset=dataset, split=split, base=base)

    def __getitem__(self, item):
        path = self.image_paths[item]
        image = Image.open(path).convert("RGB")
        image = transforms.ToTensor()(image)[:3, :, :]
        image = self.transform(image)
        image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
        # TODO: concate and return??
        answer = load_answer(self.answer_paths[item])
        return image.unsqueeze(0), answer

    def __len__(self):
        return len(self.image_paths)
