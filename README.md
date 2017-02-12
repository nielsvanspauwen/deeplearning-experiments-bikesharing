# deeplearning-experiments-bikesharing
A deep learning experiment to predict the number of bikeshare users on a given day.
I'm using a simple two-layer neural network (hand-coded for practice). The hyperparameters have been
choosen through a bit of trial and error.

The goal is to predict the number of shares a few days ahead of time, so the shop can make sure to have
the right number of bikes and staff ready to handle the load.

## Possible improvements
1. Don't use a constant learning rate. Start from e.g. 0.1 and gradually decrease the learning rate for every epoch, or step to a lower learning rate every 200 epochs etc. This will make the model converge more gradually towards the end, instead of constantly overshooting the local minima.

2. The dataset only has one Christmas period in it, and as a result, the model is not very accurate for this period. More data would help to improve the model.

3. Use the hourly data to predict expected load for the rest of the day, so shop owner can e.g. send some of the staff home in the afternoon.

