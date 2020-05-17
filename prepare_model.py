## PREPARE MODEL FOR LYRICS GENERATION:

import os
import torch
import logging
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm, trange

import utils.utilities as U

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reload the model and the tokenizer
model = GPT2LMHeadModel.from_pretrained(LOAD_MODEL_DIR)
enc = GPT2Tokenizer.from_pretrained(LOAD_MODEL_DIR)

model.to(device)
model.eval()
