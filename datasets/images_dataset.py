from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, source_root, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.source_transform = source_transform
		self.label_nc = 0

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.label_nc == 0 else from_im.convert('L')

		if self.source_transform:
			from_im = self.source_transform(from_im)

		return from_im
