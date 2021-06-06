using Flux, Plots, Statistics
using Flux.Optimise:update!
using Flux.Data:DataLoader

include("lkv.jl")
include("latent_ode.jl")


lode = new_latent_ode(2, 16, 8)

lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...; train=true)
lkv_test_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...)
nsamples = 32

lkv_train_tuple = (
    Float32.(Flux.stack(
        [
            Flux.stack(lkv_train_samples.u[i:i - 1 + nsamples], 2)
            for i in 1:length(lkv_train_samples) + 1 - nsamples
        ],
        3
    )), # 2 x 16 x numsamples
    Float32.(Flux.stack(
        [
            lkv_train_samples.t[i:i - 1 + nsamples]
            for i in 1:length(lkv_train_samples.t) + 1 - nsamples
        ],
        2
    ))
)

train_loader = DataLoader(lkv_train_tuple; batchsize=8, shuffle=true)

θ = Flux.params(lode.dzdt, lode.vae.encoder.rnn_network, lode.vae.decoder.network)
opt = ADAM(1e-5)
for i in 1:800
    display("[Epoch $(i)]")
    epoch_loss = 0
    for (x, t) in train_loader
        (x, t) = (x, t)
        grad = gradient(θ) do
            batchsize = size(t)[end]
            elbo_losses = [elbo(lode, x[:, :, i], t[:, i]) for i in 1:batchsize]
            elbo_losses = elbo(lode, x, t)
            loss = mean(elbo_losses)
            # TODO should we add reg?
            epoch_loss += loss * batchsize / length(train_loader)
            return loss
        end
        update!(opt, θ, grad)
    end
    display("Loss: $(epoch_loss)")
end

lode = lode |> cpu

lode_sol = lode(
    Float32.(Flux.stack(lkv_train_samples.u, 2)),
    Float32.(lkv_train_samples.t),
    (0, 100)
)

p = plot(
    LinRange(0, 100, 500), Flux.stack(lode_sol.(LinRange(0, 100, 500)), 1)
)