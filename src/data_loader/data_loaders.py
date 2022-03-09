from torchvision import transforms
from base import BaseDataLoader
from data_loader.data import benchmark_data


class DataLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True,  num_workers=1, task=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        # ])
        self.data_dir = data_dir
        self.dataset = benchmark_data(self.data_dir, task=task, transform=None)
        validation_split = 0.0
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


