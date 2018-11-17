# image_classification
Machine learning algorithm tryout to classify images.

## Setup

`pip install -r requirements.txt`

## Prep your datasets

Your datasets should be saved in `data` directory as a folder, let's say `data/ds`
In `ds` folder you should have `dsTrain.csv`, `dsVal.csv`, and `dsTrain.csv`

## Getting Started

Run `python -m main {train,predict} DATASET -a {dt,nb,rf}`

- `train`: train and validate with your train set and validation set
- `predict`: predict with your test set
- `DATASET`: the name of your data (data directory)
- `-a`: choose which algorithm to train/predict (Decision Tree, Naive Bayes, or Random Forest)

To tweak the hyperparameters, go to `classification.py` and manually change the values.

## Testing Scripts

ds1:

`python -m main train ds1 -a dt`

`python -m main train ds1 -a rf`

`python -m main train ds1 -a nb`

`python -m main predict ds1 -a dt`

`python -m main predict ds1 -a rf`

`python -m main predict ds1 -a nb`

ds2:

`python -m main train ds2 -a dt`

`python -m main train ds2 -a rf`

`python -m main train ds2 -a nb`

`python -m main predict ds2 -a dt`

`python -m main predict ds2 -a rf`

`python -m main predict ds2 -a nb`

