import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T

"""
    추가 작업
    Main Trainer에서 Dataloader의 Accelerator, cycle 수행해야 함.
"""


class CIFAR10Dataset(Dataset):
    """
        Args:
            root: 데이터가 저장되는 위치
            image_size: 모델에 입력하는 이미지 사이즈 (cifar-10 기본 사이즈 32*32)
            augment_horizontal_flip: horizontal flip 할 것인지,
            convert_image_to: 이미지 타입 변환 함수. 입력하지 않으면 기본 convert_image_to_fn 사용
        Usage:
            dataset = CIFAR10Dataset(...),
            dataloader = torch.utils.data.Dataloader(dataset, batch_size, shuffle, pin_memory=True, num_workers=cpu_count())
    """
    def __init__(self,
                 root,
                 image_size,
                 augment_horizontal_flip = False,
                 convert_image_to = None):
        super().__init__()
        # download cifar-10 data
        self.data = datasets.CIFAR10(root=root, train=True, download=True)

        self.image_size = image_size

        # if convert_image_to 함수가 존재하면 convert_image_to_fn에 convert_image_to 함수 부분 적용
        maybe_conver_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.transform = T.Compose([
            T.Lambda(maybe_conver_fn),
            T.Resize(image_size),   # cifar-10 image_size=32*32
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index]

        return self.transform(img)

def convert_image_to_fn(img_type, image):
    # convert image type
    if image.mode != img_type:
        return image.convert(img_type)
    return image
