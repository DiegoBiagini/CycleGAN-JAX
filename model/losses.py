import torch
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
import chex
from typing import Optional

class GANLoss(nn.Module):
    gan_mode : str
    target_real_label : Optional[float] = 1.0
    target_fake_label : Optional[float] = 0.0

    def setup(self) -> None:
        self.real_label = jnp.asarray(self.target_real_label)
        self.fake_label = jnp.asarray(self.target_fake_label)

        if self.gan_mode == 'lsgan':
            self.loss = optax.l2_loss
        elif self.gan_mode == 'vanilla':
            self.loss = optax.sigmoid_binary_cross_entropy
        elif self.gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return jnp.broadcast_to(target_tensor, prediction.shape)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -jnp.mean(prediction)
            else:
                loss = jnp.mean(prediction)
        return loss

def get_GAN_loss(gan_mode : str, target_real_label = 1.0, target_fake_label = 0.0):
    real_label = jnp.asarray(target_real_label)
    fake_label = jnp.asarray(target_fake_label)

    if gan_mode == 'lsgan':
        loss = lambda x, y : jnp.mean(optax.l2_loss(x,y))
    elif gan_mode == 'vanilla':
        loss = lambda x,y : jnp.mean(optax.sigmoid_binary_cross_entropy(x,y))
    elif gan_mode in ['wgangp']:
        loss = None
    else:
        raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    
    if gan_mode in ['lsgan', 'vanilla']:

        def loss_fn(prediction, target_is_real):
            if target_is_real:
                target_tensor = real_label
            else:
                target_tensor = fake_label
            target_tensor = jnp.broadcast_to(target_tensor, prediction.shape)
            loss_out = loss(prediction, target_tensor)
            return loss_out

    elif gan_mode == 'wgangp':
        def loss_fn(prediction, target_is_real):
            if target_is_real:
                loss_out = -jnp.mean(prediction)
            else:
                loss_out = jnp.mean(prediction)
            return loss_out

    return loss_fn

def l1_loss( predictions: chex.Array, targets: Optional[chex.Array] = None,) -> chex.Array:
    """Calculates the L1 loss for a set of predictions.
    """
    chex.assert_type([predictions], float)
    if targets is not None:
        # Avoid broadcasting logic for "-" operator.
        chex.assert_equal_shape((predictions, targets))
    errors = (predictions - targets) if (targets is not None) else predictions
    return jnp.mean(jnp.abs(errors))