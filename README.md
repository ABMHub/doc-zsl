# Document Zero-Shot Matching

To install the required libraries:

```
pip install -r requirements.txt
```

To train a model, run the `src/main.py` file. Follow the comments on the code to custom the training setup.

This project uses [WandB](wandb.ai) to track results. To run the main.py file, either setup a wandb account or set `wandb_flag` to `False`

## Project Structure

`architecture.py` holds every model implementation used on this project, implementing the model construction and its forward pass. Aside from the architecture itself, each model class contains their respective training hyperparameters.

`dataloader.py` implements every logic required to load a image, properly pre-process it and build a proper data batch.

`trainer.py` is where the training loop itself is written.

The `main.py` and the `tuning_main.py` files are both used to start the training, except, `main.py` starts only one training, and `tuning_main.py` uses the sweep funcitonality of wandb to create a hypeparameter tuning training. It is also used to automatically administrate a k-fold cross-validation.