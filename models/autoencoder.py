import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np

import sys
# sys.path.insert(0, '../../..')  # Add the parent directory to the search path
sys.path.insert(0, '../..')  # Add the parent directory to the search path
sys.path.insert(0, '..')  # Add the parent directory to the search path
sys.path.insert(0, '.')  # Add the parent directory to the search path
from bivaecf.recom_bivaecf import BiVAECF
from bivaecf.dataset import Dataset
from torch.utils.data import DataLoader


