# Document Zero-Shot Matching

To install the required libraries:

```
pip install -r requirements.txt
```

To train a model, run the `src/tuning_main.py` file. Follow the comments on the code to custom the training setup.

This project uses [WandB](wandb.ai) to track results. To run the main.py file, either setup a wandb account or set `wandb_flag` to `False`

## Project Structure

`architecture.py` holds every model implementation used on this project, implementing the model construction and its forward pass. Aside from the architecture itself, each model class contains their respective training hyperparameters.

`dataloader.py` implements every logic required to load a image, properly pre-process it and build a proper data batch.

`trainer.py` is where the training loop itself is written.

The `main.py` and the `tuning_main.py` files are both used to start the training, except, `main.py` starts only one training, and `tuning_main.py` uses the sweep funcitonality of wandb to create a hypeparameter tuning training. It is also used to automatically administrate a k-fold cross-validation.

﻿## Model Training Guide:
First, generate the dataset for training with [ZSL helper script's copy_flat.py.](https://github.com/Lyra1334/ZSL-Helper-Scripts) If your dataset is already good to go, ignore this step.
Then, change the `loopX` sections in the files `src/tuning_main.py`, `dataset/generate_splits.py`, `dataset/generate_protocol.py` and `dataset/separate_test.py`. Don't forget to also create a folder in `dataset/active_labeling` for the current loop and to also create the `protocols` and `splits` folder inside of it.
Then, you run, in order, `dataset/generate_splits.py`, `dataset/generate_protocol.py`, and `dataset/separate_test.py`. Some of these scripts may fail at random, so I don't recommend writing a .sh file to run them.

The, you have to comment line 219 and uncomment lines 216 and 217 and run `src/tuning_main.py`. Doing this will return you the wandb code you have to substitute in lines 219 and 45. After that, you have to comment lines 216 and 217 and uncomment line 219 and run `src/tuning_main.py` again. As running the script may take more than a day, if you are running the script on an external server, i recommend using the command `nohup python3 src/tuning_main.py &> /dev/null &`, that lets the process remain running unattached to your terminal and doesn't generate the gigantic log file wandb would produce without the `&> /dev/null/ &` argument.
