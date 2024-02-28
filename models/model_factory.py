import torch
import logging



def build_model(cfg):
	if cfg.model.name == "vae":
		model = VAE()
		return model
		

			
			
			
 