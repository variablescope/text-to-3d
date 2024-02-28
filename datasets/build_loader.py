import torch
from .coco import CoCo
from .transforms import get_transform
from torchvision.datasets import ImageFolder


def build_loader(cfg):
	if cfg.dataset.name == "coco":
		train_ds = CoCo(cfg, dataType='train2017', annType='captions', is_train=True)
		
	
	train_dl = torch.utils.data.DataLoader(train_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers)  
	val_dl = torch.utils.data.DataLoader(val_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers) 
	
	return (train_dl, val_dl)


			
			
 