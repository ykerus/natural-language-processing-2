
from torch.utils.data import Dataset
from data.utils import *

class CustomDataset(Dataset):
  """Shapes varying in position and size"""

  def __init__(self, shape, size=28, num_samples=500, dynamic=True, distractors=0):

    self.shape = shape
    self.size = size
    self.num_samples = num_samples
    self.images, self.targets = generate_data(shape=shape, size=size, num_samples=num_samples)
    self.dynamic = dynamic
    if not dynamic:
      self.ref_atts = alter_attributes(shape, self.targets)
      self.reference = reconstruct(shape, self.ref_atts)
      
    if distractors >= 5:
      if dynamic:
        print("Recommended: use a static dataset, or lower the number the distractors")
      self.distractors = get_distractors(distractors, shape, self.targets, self.ref_atts, size=size)
    else:
      self.distractors = torch.zeros((num_samples, distractors, 1, 28, 28))
      for i in range(distractors):
        self.distractors[:,i,:,:,:] = reconstruct(shape, alter_attributes(shape, self.targets))
         
  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.dynamic:
      return (self.images[idx], self.targets[idx], self.targets[idx], self.distractors[idx])
    else:
      return (self.images[idx], self.targets[idx], self.reference[idx], self.distractors[idx], self.ref_atts[idx])