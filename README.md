# KISS
Code for the paper [KISS: Keeping it Simple for Scene Text Recognition](https://arxiv.org/abs/1911.08400).

This repository contains the code you can use in order to train a model based on our paper.
You will also find instructions on how to access our model and also how to evaluate the model.

# Pretrained Model

You can find the pretrained model [here](https://bartzi.de/research/kiss).
Download the zip and put into any directory. We will refer to this
directory as `<model_dir>`.

# Prepare for using the Code

- make sure you have at least Python **3.7** installed on your system
- create a new virtual environment (or whatever you like to use)
- install all requirements with `pip install -r requirements.txt`
(if you do not have a CUDA capable device in your PC, you should remove
the package `cupy` from the file `requirements.txt`).

# Datasets

If you want to train your model on the same datasets, as we did, you'll 
need to get the train data first. Second, you can get the train annotation
we used from [here](https://bartzi.de/research/kiss).

## Image Data

You can find the image data for each dataset, using the following links:
- MJSynth: https://www.robots.ox.ac.uk/~vgg/data/text/ 
- SynthText: https://www.robots.ox.ac.uk/~vgg/data/scenetext/
- SynthAdd: Follow instructions from [here](https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition)

Once, you've downloaded all the images, you can get the gt-files we've prepared for
the MJSynth and SynthAdd datasets [here](https://bartzi.de/research/kiss).

For the SynthText dataset, you'll have to create them yourself.
You can do so by following these steps:
1. Get the data and put it into a directory (lets assume we put the data into the
directory `/data/oxford`)
1. run the script `crop_words_from_oxford.py` (you can find it in `datasets/text_recognition`)
with the following command line parameters `python crop_words_from_oxford.py /data/oxford/gt.mat /data/oxford_words`.
1. This will crop all words based on their axis aligned bounding box from the
original oxford gt.
1. Create train and validation split with the script `create_train_val_splits.py`.
`python create_train_val_splits.py /data/oxford_words/gt.json`.
1. Run the script `json_to_npz.py` with the following command line:
`python json_to_npz /data/oxford_words/train.json ../../train_utils/char-map-bos.json`.
This will create a file called `train.npz` in the same directory as the file `gt.json` is currently located in.
1. Repeat the last step with the files `validation.json`.

Once you are done with this, you'll need to combine all `npz` files into
one large `npz` file. You can use the `combine_npz_datasets.py` for this.
Assume you saved the MJSynth dataset + npz file here `/data/mjsynth` and
the SynthAdd dataset + npz file here `/data/SynthAdd`, then you'll need
to run the script in the following way: `python combine_npz_datasets.py 
/data/mjsynth/annotation_train.npz /data/oxford_words/train.npz /data/SynthAdd/gt.npz
--destination /data/datasets_combined.npz`.

Since the datasets may contain words that are longer than `N` characters (we always set `N` to 23),
we need to get rid of all words that are longer than `N` characters.
You can use the script `filter_word_length.py` for this.
Use it like so: `python filter_word_length.py 23 /data/datasets_combined.npz --npz`.
Do the same thing with the file `validation.npz` you obtained from splitting
the SynthText dataset.

If you want to follow our experiments with the balanced dataset, you can 
create a balanced dataset with the script `balance_dataset.py`.
For example: `python balance_dataset.py /data/datasets_combined_filtered_23.npz datasets_combined_balanced_23.npz -m 200000`.
If you do not use the `-m` switch the script will show you dataset statistics and you can choose your own value.

## Evaluation Data

In this ssection we explain, hou you can get the evaluation data + annotation.
For getting the evaluation data you just need to do 2 steps per dataset:
1. Clone the repository.
1. Download the `npz` annotation file. And place it in the directory, where you cloned the git repository to.

| Dataset  | Git Repo | NPZ-Link | Note |
|---|---|---|---|
| ICDAR2013 | https://github.com/ocr-algorithm-and-data/ICDAR2013 | [download](https://bartzi.de/documents/attachment/download?hash_value=4e1c652bf62fb2e454cb65c3d996f592_30) | Rename the directory `test` to `Challenge2_Test_Task3_Images` |
| ICDAR2015 | https://github.com/ocr-algorithm-and-data/ICDAR2015 | [download](https://bartzi.de/documents/attachment/download?hash_value=8fdafd36ce2e1108913081fd247e54b4_31) | Rename the dir `TestSet` to `ch4_test_word_images_gt` |
| CUTE80 | https://github.com/ocr-algorithm-and-data/CUTE80 | [download](https://bartzi.de/documents/attachment/download?hash_value=340f042991cf752d0ba6c700afa1bdb0_28) | - |
| IIIT5K | https://github.com/ocr-algorithm-and-data/IIIT5K | [download](https://bartzi.de/documents/attachment/download?hash_value=af4799078f54a0138d43010563301d8b_32) | - |
| SVT | https://github.com/ocr-algorithm-and-data/SVT | [download](https://bartzi.de/documents/attachment/download?hash_value=f7b99cc79d9b5bcecb1e755f4b5b1038_33) | Remove all subdirs, but the dir `test_crop`. Rename this dir to `img` |
| SVTP | https://github.com/ocr-algorithm-and-data/SVT-Perspective | [download](https://bartzi.de/documents/attachment/download?hash_value=36238a96473f2a604f8bf4c6874de55c_34) | - |

# Training
 
Now you should be ready for training :tada:.
You can use the script `train_text_recognition.py`, which is in the 
root-directory of this repo.

Before you can start your training, you'll need to adapt the config in 
`config.cfg`.
Set the values following this list:
- **train_file**: Set this to the file `/data/datasets_combined_filtered_23.npz`
- **val_file**: Set this to `/data/oxford_words.validation.npz`
- **keys in TEST_DATASETS** set those to the corresponding npz file you got [here](https://bartzi.de/research/kiss) and setup in the last step.

You can now run the training with, e.g.,
`python train_text_recognition.py <name for the log> -g 0 -l tests --image-mode RGB --rdr 0.95`
This will start the training and create a new directlry with log entries in `logs/tests`.
Get some coffee and sleep, because the training will take some time!

You can inspect the train progress with Tensorboard. Just start Tensorboard
in the root directory like so: `tensorboard --logir logs`.

# Evaluation

Once, you've trained a model or if you just downloaded the model we provided,
you can run the evaluation script on it.

If you want to know how the model performes on all datasets, you can use the 
script `run_eval_on_all_datasets.py`. Lets assume you trained a model and 
`logs/tests/train` is the path to the log dir.
Now, you can run the evaluation with this command: `python run_eval_on_all_datasets.py 
config.cfg 0 -b 16 --snapshot-dir logs/tests/train`.
You can also render the predictions of the model for each evaluation image
by making the following changes to the command:
`python run_eval_on_all_datasets.py config.cfg 0 -b 1
--snapshot-dir logs/tests/train --render`.
You will then find the results for each image in the directory `logs/tests/train/eval_bboxes`.

# Questions?

Feel free to open an issue!
You want to contribute? Just open a PR :smile:!

# License

This code is licensed under GPLv3, see the file `LICENSE` for more information.

# Citation

If you find this code useful, please cite our paper:
```bibtex
@misc{bartz2019kiss,
    title={KISS: Keeping It Simple for Scene Text Recognition},
    author={Christian Bartz and Joseph Bethge and Haojin Yang and Christoph Meinel},
    year={2019},
    eprint={1911.08400},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
