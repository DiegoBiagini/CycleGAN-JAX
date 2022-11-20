import time
from options import TrainOptions
from data.dataset_loader import DatasetDataLoader
from model.cyclegan_model import CycleGan
import jax
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    rng = jax.random.PRNGKey(0)

    rng, rng1 = jax.random.split(rng)

    dataset = DatasetDataLoader(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Register dataset size
    opt.ds_size = dataset_size
    # Computing number of steps
    opt.n_steps = opt.n_epochs*(dataset_size//opt.batch_size)
    opt.n_steps_decay = opt.n_epochs_decay*(dataset_size//opt.batch_size)

    model = CycleGan(opt, rng)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        epoch_losses = {k:[] for k in model.get_losses()}
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            iter_data_time = time.time()

            losses = model.get_losses()
            if epoch_iter % 100:
                print("Losses:" + ','.join([str(k)+ ":"+str(losses[k]) for k in losses]))
            for k in losses:
                epoch_losses[k].append(losses[k])

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        print("Average epoch losses:" + ','.join([str(k)+":"+str(np.mean(epoch_losses[k]))]))