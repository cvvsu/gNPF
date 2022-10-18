# gNPF

PyTorch code for generating particle number size distribution (PNSD) datasets.

For datasets, you can use the code [get_datasets](https://github.com/cvvsu/maskNPF/blob/main/utils/get_datasets.py) to download the measured PNSD for SMEAR I, II, and III from [https://smear.avaa.csc.fi/](https://smear.avaa.csc.fi/).

You can also use the datasets we have already downloaded:

[measured PNSD for SMEAR I, II, III](https://github.com/cvvsu/gNPF/releases/tag/v0.0)

Put the datasets under the folder `datasets` and rename them with the names `var`, `hyy`, and `kum` for SMEAR I, II, and III, respectively.

## Data Augmentation

Though we have observations for many years, we need more samples for training.
For data augmentation, please run (use `hyy` and `kum` for SMEAR II and III, respectively)

```
$ python3 datasets/obtain_csv.py --station var
```

## Run the model

We use the WGAN-GP as the default model, which can alleviate the mode collapse problem.

```
$ python3 demo.py --exp_name var --station var --epochs 20 --dataroot datasets --batch_size 64 --lr 0.0001 --lambda_gp 10 
```

To visualize the generated results, please refer to the `demo.ipynb` file for more details.

