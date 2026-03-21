import sys
sys.path.insert(0, '.')
from model import models
from src import dataset
from unlearn_strategies import unlearn
from torch.utils.data import DataLoader
import torch

train,test,ncls,nch = dataset.get_dataset('MNist', root='./data')
print('Dataset loaded, train len', len(train))
x,y = train[0]
# If dataset provides single-channel images (MNist) but model expects 3 channels, expand channels to 3 for a smoke check
if hasattr(x, 'shape') and x.shape[0] == 1:
    x = x.repeat(3, 1, 1)
# If image size is 28x28 (MNist) but model expects 32x32, pad to 32x32
import torch.nn.functional as F
if x.shape[1] == 28:
    x = F.pad(x, (2,2,2,2))
print('Single sample shapes:', getattr(x,'shape',None), y)
loader = DataLoader([(x,y)], batch_size=1)
student = models.SimpleCNN(num_classes=ncls)
teacher = models.SimpleCNN(num_classes=ncls)
opt = torch.optim.Adam(student.parameters(), lr=1e-4)
print('Running single unlearning_step...')
loss = unlearn.unlearning_step(model=student, unlearning_teacher=teacher, full_trained_teacher=teacher, unlearn_data_loader=loader, optimizer=opt, device='cpu', KL_temperature=1.0)
print('Loss:', loss)
