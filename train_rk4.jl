using CUDA, Flux, Plots, Statistics
using Flux.Optimise:update!
using Flux.Data:DataLoader


include("rk4.jl")
include("lkv.jl")

lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_samples = sample_lotka_volterra(lkv_ic, 50, 100., true_params...; train=true)
lkv_test_samples = sample_lotka_volterra(lkv_ic, 50, 100., true_params...)

lkv_train_tuple = (
    lkv_train_samples[1:end - 1],
    lkv_train_samples[2:end],
    lkv_train_samples.t[2:end] .- lkv_train_samples.t[1:end - 1]
)

train_loader = DataLoader(lkv_train_tuple; batchsize=16, shuffle=true)
test_loader = DataLoader(lkv_train_samples; batchsize=32, shuffle=false)

est_params = rand(4)
θ = params(est_params)

f(x, α, β, δ, γ) = [
    α * x[1] - β * x[1] * x[2],
    δ * x[1] * x[2] - γ * x[2]
]

f(x) = f(x, est_params...)
mse_loss(x̂, x) = mean(sum((x̂ .- x).^2, dims=2))

opt = Descent(0.1)
for i in 1:10
    for (x₀, xₜ, T) in train_loader
        t = LinRange.(0, T, 100)
        x̂ₜ = collect(map((t, x) -> rk4(f, t, x), t, eachrow(x₀)))
        grad = gradient(mse_loss(x̂ₜ, xₜ), θ)
        update!(opt, θ, grad)
    end
end

fc_model = Chain(
    Dense(2, 16, relu),
    Dense(16, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 2)
)
