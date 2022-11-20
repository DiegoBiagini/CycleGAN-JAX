import functools
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import jax.numpy as jnp
import optax

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm, use_bias=True, use_running_average=True)
    elif norm_type == 'instance':
        raise NotImplementedError("Instance norm is not implemented")
        #norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return x
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        scheduler = optax.polynomial_schedule(opt.lr, 0, 1, opt.n_steps_decay, opt.n_steps)
    elif opt.lr_policy == 'step':
        boundaries = {v:0.1 for v in [jnp.arange(1,opt.n_steps, opt.lr_decay_iters*(opt.ds_size // opt.batch_size))]}
        scheduler = optax.piecewise_constant_schedule(opt.lr, boundaries_and_scales=boundaries)
    elif opt.lr_policy == 'cosine':
        scheduler = optax.cosine_decay_schedule(opt.lr, opt.n_steps)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_init_strategy(init_type : str ='normal'):
    init_strategy = {}
    if init_type == 'normal':
        init_strategy["Conv"] = jax.nn.initializers.normal(stddev=0.02)
        init_strategy["Linear"] = jax.nn.initializers.normal(stddev=0.02)
    elif init_type == 'xavier':
        init_strategy["Conv"] = jax.nn.initializers.glorot_normal()
        init_strategy["Linear"] = jax.nn.initializers.glorot_normal()
    elif init_type == 'kaiming':
        init_strategy["Conv"] = jax.nn.initializers.kaiming_normal()
        init_strategy["Linear"] = jax.nn.initializers.kaiming_normal()
    elif init_type == 'orthogonal':
        init_strategy["Conv"] = jax.nn.initializers.orthogonal()
        init_strategy["Linear"] = jax.nn.initializers.orthogonal()
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    init_strategy["Bias"] = jax.nn.initializers.constant(0)
    return init_strategy


def init_net(net, rng, dummy_shape):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    rng1, rng2 = jax.random.split(rng)
    x = random.normal(rng1, dummy_shape) # Dummy input
    params = net.init(rng2, x) # Initialization cal
    #print(net.tabulate(rng1,x))
    return params


def define_G(output_nc, ngf, rng, dummy_shape, norm='batch', use_dropout=False, init_type='normal'):
    """Create a generator

    Parameters:
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    init_strategy = define_init_strategy(init_type=init_type)

    net = ResnetGenerator( output_nc=output_nc, ngf=ngf, 
                            norm_layer=norm_layer, use_dropout=use_dropout, 
                            n_blocks=9, init_strategy=init_strategy)
    params = init_net(net, rng, dummy_shape)
    return net, params


def define_D(ndf, rng, dummy_shape, norm='batch', init_type='normal'):
    """Create a discriminator

    Parameters:
        ndf (int)          -- the number of filters in the first conv layer
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    init_strategy = define_init_strategy(init_type=init_type)

    # default PatchGAN classifier
    net = NLayerDiscriminator(ndf, n_layers=3, norm_layer=norm_layer, init_strategy=init_strategy)
    params = init_net(net, rng, dummy_shape)
    return net, params

class ResnetGenerator(nn.Module):
    output_nc : int
    ngf : Optional[int] = 64
    norm_layer : Optional[callable] = nn.BatchNorm
    use_dropout : Optional[bool] = False
    n_blocks : Optional[int] = 6
    padding_type : Optional[str] = 'reflect'
    init_strategy : Optional[callable] = None

    def setup(self) -> None:
        assert(self.n_blocks >= 0)

        """
        # TODO: After implementation of instance norm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """
        if self.init_strategy is None:
            init_conv_w = jax.nn.initializers.lecun_normal()
            init_conv_b = jax.nn.initializers.zeros()
        else:
            init_conv_w = self.init_strategy["Conv"]
            init_conv_b = self.init_strategy["Bias"]

        use_bias = False
        
        self.reflection_padder = lambda x : jnp.pad(x, [(0,0),(3,3),(3,3),(0,0)], mode="reflect")

        model = [self.reflection_padder,
                 nn.Conv(features=self.ngf, kernel_size=(7,7), padding="VALID", 
                        use_bias=use_bias, kernel_init=init_conv_w, bias_init=init_conv_b),
                 self.norm_layer(),
                 nn.relu]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv(features=self.ngf*mult*2, kernel_size=(3,3), strides=(2,2), 
                        padding=(1,1), use_bias=use_bias, kernel_init=init_conv_w, bias_init=init_conv_b),
                      self.norm_layer(),
                      nn.relu]

        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):       # add ResNet blocks
            model += [ResnetBlock(self.ngf * mult, padding_type=self.padding_type, 
                        norm_layer=self.norm_layer, use_dropout=self.use_dropout, 
                        use_bias=use_bias, kernel_init=init_conv_w, bias_init=init_conv_b)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose(features=int(self.ngf * mult / 2), kernel_size=(2,2),
                                        strides=(2,2), padding="VALID", use_bias=use_bias, kernel_init=init_conv_w, bias_init=init_conv_b),
                      self.norm_layer(),
                      nn.relu]
        model += [self.reflection_padder]
        model += [nn.Conv(features=self.output_nc, kernel_size=(7,7), padding="VALID", kernel_init=init_conv_w, bias_init=init_conv_b)]
        model += [nn.tanh]

        self.model = nn.Sequential(model)

    def __call__(self, input) -> Any:
        return self.model(input)

class ResnetBlock(nn.Module):
    dim : int 
    padding_type : str
    norm_layer : nn.Module
    use_dropout : bool
    use_bias : bool
    kernel_init : callable 
    bias_init : callable

    def setup(self) -> None:
        
        if self.padding_type == 'reflect':
            self.padder =  lambda x : jnp.pad(x, [(0,0),(1,1),(1,1),(0,0)], mode="reflect")
        elif self.padding_type == 'replicate':
            self.padder =  lambda x : jnp.pad(x, [(0,0),(1,1),(1,1),(0,0)], mode="edge")
        elif self.padding_type == 'zero':
            self.padder = lambda x : jnp.pad(x, [(0,0),(1,1),(1,1),(0,0)], mode="constat")
        else:
            raise NotImplementedError('padding [%s] is not implemented' % self.padding_type)

        conv_block = []
        conv_block += [self.padder]
        conv_block += [nn.Conv(features=self.dim, kernel_size=(3,3), padding="VALID", 
                        use_bias=self.use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init),
                        self.norm_layer(), 
                        nn.relu]
        if self.use_dropout:
            conv_block += [nn.Dropout(rate=0.5)]
        conv_block += [self.padder]
        conv_block += [nn.Conv(features=self.dim, kernel_size=(3,3), padding="VALID", use_bias=self.use_bias, 
                                kernel_init=self.kernel_init, bias_init=self.bias_init),
                        self.norm_layer(), 
                        nn.relu]

        self.conv_block = nn.Sequential(conv_block)

    def __call__(self, x) -> Any:
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module):
    ndf : Optional[int] = 64
    n_layers : Optional[int] = 3
    norm_layer : Optional[callable] = nn.BatchNorm
    init_strategy : Optional[callable] = None

    def setup(self) -> None:
        """
        # TODO: After implementation of instance norm

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """
        if self.init_strategy is None:
            init_conv_w = jax.nn.initializers.lecun_normal()
            init_conv_b = jax.nn.initializers.zeros()
        else:
            init_conv_w = self.init_strategy["Conv"]
            init_conv_b = self.init_strategy["Bias"]

        use_bias = False

        kw = 4
        padw = 1
        sequence = [nn.Conv(self.ndf, kernel_size=(kw,kw), strides=(2,2), padding=(padw,padw), kernel_init=init_conv_w, bias_init=init_conv_b), 
                    functools.partial(nn.leaky_relu, negative_slope=0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv(self.ndf*nf_mult, kernel_size=(kw,kw), strides=(2,2), padding=(padw,padw), use_bias=use_bias, kernel_init=init_conv_w, bias_init=init_conv_b),
                self.norm_layer(),
                functools.partial(nn.leaky_relu, negative_slope=0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.n_layers, 8)
        sequence += [
            nn.Conv(self.ndf*nf_mult, kernel_size=(kw,kw), strides=(1,1), padding=(padw,padw), use_bias=use_bias, kernel_init=init_conv_w, bias_init=init_conv_b),
            self.norm_layer(),
            functools.partial(nn.leaky_relu, negative_slope=0.2)
        ]

        sequence += [nn.Conv(1, kernel_size=(kw,kw), strides=(1,1), padding=(padw,padw), kernel_init=init_conv_w, bias_init=init_conv_b)]  # output 1 channel prediction map
        self.model = nn.Sequential(sequence)

    def __call__(self, input) -> Any:
        return self.model(input)
