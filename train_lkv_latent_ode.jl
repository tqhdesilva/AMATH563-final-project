using Flux, Plots, Statistics
using Flux.Optimise:update!
using Flux.Data:DataLoader

include("lkv.jl")
include("latent_ode.jl")


lode = LatentODE(2, 16, 16)

lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...; train=true)
lkv_test_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...)


lkv_train_tuple = (
    Float32.(Flux.stack(
        [
            Flux.stack(lkv_train_samples.u[i:i + 16 - 1], 2)
            for i in 1:length(lkv_train_samples) - 16
        ],
        3
    )), # 2 x 16 x numsamples
    Float32.(Flux.stack(
        [
            lkv_train_samples.t[i:i + 16 - 1]
            for i in 1:length(lkv_train_samples.t) - 16
        ],
        2
    )) # 16 x numsamples
)

train_loader = DataLoader(lkv_train_tuple; batchsize=16, shuffle=true)

θ = Flux.params(lode.dzdt, lode.vae.encoder.rnn_network, lode.vae.decoder.network)
opt = Descent(0.0001)

for i in 1:10
    display("[Epoch $(i)]")
    for (x, t) in train_loader
        display("training")
        grad = gradient(θ) do
            batchsize = size(t)[end]
            elbo_losses = [elbo(lode, x[:, :, i], t[:, i]) for i in 1:batchsize]
            loss = mean(elbo_losses)
            # TODO should we add reg?
            return loss
        end
        update!(opt, θ, grad)
    end
end