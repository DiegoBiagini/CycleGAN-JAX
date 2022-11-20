import os
from .networks import define_G, define_D, get_scheduler
from .image_pool import ImagePool
from .losses import get_GAN_loss, l1_loss
from collections import OrderedDict
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any
import numpy as np

class TrainState(train_state.TrainState):
  batch_stats: Any

class CycleGan():

    def __init__(self, opt, rng):
        """Initialize the CycleGan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            self.losses = {
                "D_A":None,
                "G_A":None,
                "D_B":None,
                "G_B":None
            }
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # Input shapes, for network initialization
        dummy_shape = (1,opt.crop_size, opt.crop_size, opt.input_nc)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        rng, rng1 = jax.random.split(rng)
        self.netG_A, self.parG_A = define_G(opt.output_nc, opt.ngf, rng1, dummy_shape, opt.norm,
                                        not opt.no_dropout, opt.init_type,)
        self.netG_B, self.parG_B = define_G(opt.output_nc, opt.ngf, rng1, dummy_shape, opt.norm,
                                        not opt.no_dropout, opt.init_type)

        if self.isTrain:  # define discriminators
            rng, rng1 = jax.random.split(rng)
            self.netD_A, self.parD_A = define_D(opt.ndf, rng1, dummy_shape, opt.norm, opt.init_type)

            rng, rng1 = jax.random.split(rng)
            self.netD_B, self.parD_B = define_D(opt.ndf, rng1, dummy_shape, opt.norm, opt.init_type)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = get_GAN_loss(opt.gan_mode) # define GAN loss.
            self.criterionCycle = l1_loss
            self.criterionIdt = l1_loss

            # initialize optimizers
            # Define schedules
            schedule = get_scheduler(opt)

            self.optimizer_GA = optax.adam(learning_rate=schedule, b1=opt.beta1)
            self.optimizer_DA = optax.adam(learning_rate=schedule, b1=opt.beta1)
            self.optimizer_GB = optax.adam(learning_rate=schedule, b1=opt.beta1)
            self.optimizer_DB = optax.adam(learning_rate=schedule, b1=opt.beta1)

            self.optimizer_GA_state = TrainState.create(apply_fn=self.netG_A.apply, params=self.parG_A['params'], 
                                                        batch_stats=self.parG_A['batch_stats'], tx=self.optimizer_GA)
            self.optimizer_GB_state = TrainState.create(apply_fn=self.netG_B.apply, params=self.parG_B['params'], 
                                                        batch_stats=self.parG_B['batch_stats'],  tx=self.optimizer_GB)
            self.optimizer_DA_state = TrainState.create(apply_fn=self.netD_A.apply, params=self.parD_A['params'], 
                                                        batch_stats=self.parD_A['batch_stats'], tx=self.optimizer_DA)
            self.optimizer_DB_state = TrainState.create(apply_fn=self.netD_B.apply, params=self.parD_A['params'], 
                                                        batch_stats=self.parD_B['batch_stats'], tx=self.optimizer_DB)

            self.optimizers.append(self.optimizer_GA)
            self.optimizers.append(self.optimizer_DA)
            self.optimizers.append(self.optimizer_GB)
            self.optimizers.append(self.optimizer_DB)

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if not self.isTrain:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            # TODO:
            self.load_networks(load_suffix)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = jnp.moveaxis(input['A' if AtoB else 'B'].numpy(), 1, 3)
        self.real_B = jnp.moveaxis(input['B' if AtoB else 'A'].numpy(), 1, 3)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_losses(self):
        return self.losses

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.generator_step()
        self.discriminator_step()
    
    def discriminator_step(self):
        # Get pooled fake samples
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_A = self.fake_A_pool.query(self.fake_A)

        # Discriminator A
        def loss_fn_DA(params):
            # Real
            pred_real, mutables = self.optimizer_DA_state.apply_fn(
                {'params': params, 'batch_stats': self.optimizer_DA_state.batch_stats},
                self.real_B, mutable=['batch_stats'])
            loss_DA_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake, mutables = self.optimizer_DA_state.apply_fn(
                {'params': params, 'batch_stats': self.optimizer_DA_state.batch_stats},
                fake_B, mutable=['batch_stats'])
            loss_DA_fake = self.criterionGAN(pred_fake, False)
            # Combined loss
            loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
            return loss_DA, mutables

        # Critique real and fake samples
        grad_fn_A = jax.value_and_grad(loss_fn_DA, has_aux=True)
        (DA_loss, mutables), grads_DA = grad_fn_A(self.parD_A["params"])
        # Update the Discriminator A.
        self.optimizer_DA_state = self.optimizer_DA_state.apply_gradients( grads=grads_DA, batch_stats=mutables['batch_stats'])
        if self.isTrain:
            self.losses["D_A"] = DA_loss

        # Discriminator B
        def loss_fn_DB(params):
            pred_real, mutables = self.optimizer_DB_state.apply_fn(
                {'params': params, 'batch_stats': self.optimizer_DB_state.batch_stats},
                self.real_A, mutable=['batch_stats'])
            loss_DB_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake, mutables = self.optimizer_DB_state.apply_fn(
                {'params': params, 'batch_stats': self.optimizer_DB_state.batch_stats},
                fake_A, mutable=['batch_stats'])
            loss_DB_fake = self.criterionGAN(pred_fake, False)
            # Combined loss
            loss_DB = (loss_DB_real + loss_DB_fake) * 0.5
            return loss_DB, mutables

        # Critique real and fake samples
        grad_fn_B = jax.value_and_grad(loss_fn_DB, has_aux=True)
        (DB_loss, mutables), grads_DB = grad_fn_B(self.parD_B["params"])
        # Update the Discriminator B.
        self.optimizer_DB_state = self.optimizer_DB_state.apply_gradients( grads=grads_DB, batch_stats=mutables['batch_stats'])
        if self.isTrain:
            self.losses["D_B"] = DB_loss

    def generator_step(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_idt = self.opt.lambda_identity

        def loss_fn_GA(params):
            generated_data_AtoB, mutables = self.optimizer_GA_state.apply_fn(
                {'params': params, 'batch_stats': self.optimizer_GA_state.batch_stats},
                self.real_A, mutable=['batch_stats'])
            self.fake_B = generated_data_AtoB

            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                generated_data_BtoB, mutables = self.optimizer_GA_state.apply_fn(
                    {'params': params, 'batch_stats': self.optimizer_GA_state.batch_stats},
                    self.real_B, mutable=['batch_stats'])
                loss_idt_A = self.criterionIdt(generated_data_BtoB, self.real_B) * lambda_B * lambda_idt
            else:
                loss_idt_A = 0

            # GAN loss D_A(G_A(A))
            discrim_out, mutables = self.optimizer_DA_state.apply_fn(
                {'params': self.parD_A["params"], 'batch_stats': self.optimizer_DA_state.batch_stats},
                generated_data_AtoB, mutable=['batch_stats'])
            loss_G_A = self.criterionGAN(discrim_out, True)

            # Forward cycle loss || G_B(G_A(A)) - A||
            self.rec_A, mutables = self.optimizer_GB_state.apply_fn(
                {'params': self.parG_B["params"], 'batch_stats': self.optimizer_GB_state.batch_stats},
                generated_data_AtoB, mutable=['batch_stats'])
            loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

            loss = loss_G_A + loss_cycle_A + loss_idt_A
            return loss, mutables
        # Generate data with the Generator A, critique it with the Discriminator A and check cycle consistency
        grad_fn_A = jax.value_and_grad(loss_fn_GA, has_aux=True)
        (GA_loss, mutables), grads_GA = grad_fn_A(self.parG_A["params"])
        self.fake_B = np.asarray(jax.lax.stop_gradient(self.fake_B))

        # Update the Generator A.
        self.optimizer_GA_state = self.optimizer_GA_state.apply_gradients( grads=grads_GA, batch_stats=mutables['batch_stats'])
        if self.isTrain:
            self.losses["G_A"] = GA_loss
        
        def loss_fn_GB(params):
            generated_data_BtoA, mutables = self.optimizer_GB_state.apply_fn(
                {'params': params, 'batch_stats': self.optimizer_GB_state.batch_stats},
                self.real_B, mutable=['batch_stats'])
            self.fake_A = generated_data_BtoA

            # Identity loss
            if lambda_idt > 0:
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                generated_data_AtoA, mutables = self.optimizer_GB_state.apply_fn(
                    {'params': params, 'batch_stats': self.optimizer_GB_state.batch_stats},
                    self.real_A, mutable=['batch_stats'])
                loss_idt_B = self.criterionIdt(generated_data_AtoA, self.real_A) * lambda_A * lambda_idt
            else:
                loss_idt_B = 0

            # GAN loss D_B(G_B(B))
            discrim_out, mutables = self.optimizer_DB_state.apply_fn(
                {'params': self.parD_B["params"], 'batch_stats': self.optimizer_DB_state.batch_stats},
                generated_data_BtoA, mutable=['batch_stats'])
            loss_G_B = self.criterionGAN(discrim_out, True)

            # Backward cycle loss || G_A(G_B(B)) - B||
            self.rec_B, mutables = self.optimizer_GA_state.apply_fn(
                {'params': self.parG_A["params"], 'batch_stats': self.optimizer_GA_state.batch_stats},
                generated_data_BtoA, mutable=['batch_stats'])
            loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

            loss = loss_G_B + loss_cycle_B + loss_idt_B
            return loss, mutables
        
        # Generate data with the Generator B, critique it with the Discriminator B and check cycle consistency
        grad_fn_B = jax.value_and_grad(loss_fn_GB, has_aux=True)
        (GB_loss, mutables), grads_GB = grad_fn_B(self.parG_B["params"])
        self.fake_A = np.asarray(jax.lax.stop_gradient(self.fake_A))

        # Update the Generator A.
        self.optimizer_GB_state = self.optimizer_GB_state.apply_gradients( grads=grads_GB, batch_stats=mutables['batch_stats'])

        if self.isTrain:
            self.losses["G_B"] = GB_loss
