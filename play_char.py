import os
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from models.gpt import GPT, GPTConfig
from data.char_dataset import CharDataset
from trainer import Trainer, TrainerConfig

block_size = 128 # spatial extent of the model for its context

text = open('tinyshakespeare.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(text, block_size) # one line of poem is roughly 50 characters

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

tconf = TrainerConfig(max_epochs=2, batch_size=64, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                      num_workers=4, gpus=1)
print(tconf.batch_size)
trainer = Trainer(model, tconf, train_dataset, None)
trainer.train()