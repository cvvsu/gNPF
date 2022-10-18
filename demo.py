import os, time
import numpy as np 
import pandas as pd 
from options import get_args
from datasets import get_dataset
from model import build_model
from datetime import datetime
import torch


if __name__=="__main__":
    tic = time.time()    # start the program
    args, msg = get_args()
    print(msg)

    loader = get_dataset(args)
    model = build_model(args)

    model.train(loader)   
    
    print(f'Time consuming {time.time()-tic} s for whole experiment.')


         
