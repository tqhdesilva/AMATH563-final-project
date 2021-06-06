using Flux, Plots

include("lkv.jl")
include("latent_ode.jl")


lode = LatentODE(2, 16, 16)
lkv_train_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...; train=true)
lkv_test_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...)
