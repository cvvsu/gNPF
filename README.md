# gNPF

PyTorch code for generating particle number size distribution (PNSD) datasets.

## Data Augmentation

Though we have observations for many years, while we need more samples in the observed PNSD datasets.
For data augmentation, please run (use `hyy` and `kum` for SMEAR II and III, respectively)

```
$ python3 datasets/obtain_csv.py --station var
```

## Run the model

We use the WGAN-GP as the default model, which can alleviate the mode collapse problem.

```
$ python3 demo.py --exp_name var --epochs 20 --dataroot datasets --batch_size 64 --lr 0.0001 --lambda_gp 10 --station var
```

To visualize the generated results, please refer to the `demo.ipynb` file for more details.

