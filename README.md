# Cyclegan-JAX

Basic porting of the original pytorch CycleGAN implementation in Jax + Flax.  
Reference repository: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  

### What's missing
- Saving the model
- Loading the model
- Data loading/preprocessing is entirely in pytorch and I much prefer that to switching to tf
- Flax ConvTranspose does not have an output_padding parameter, so I changed the generator architecture slightly
- The default settings used InstanceNorm but Flax has no InstanceNorm
- Maybe investigate jitting 
- Dropout is disabled for now, but enabling it would probably require some changes
