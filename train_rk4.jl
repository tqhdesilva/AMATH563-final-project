## Create datasets, loaders
using CUDA, Flux, Plots, Statistics
using Flux.Optimise:update!
using Flux.Data:DataLoader


include("rk4.jl")
include("lkv.jl")

lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...; train=true)
lkv_test_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...)

lkv_train_tuple = (
    lkv_train_samples[:, 1:end - 1],
    lkv_train_samples[:, 2:end],
    lkv_train_samples.t[2:end] .- lkv_train_samples.t[1:end - 1]
)

train_loader = DataLoader(lkv_train_tuple; batchsize=16, shuffle=true)
test_loader = DataLoader(lkv_train_samples; batchsize=32, shuffle=false)

## Train rk4 model to learn lkv params
est_params = rand(4)
println(est_params)
θ = Flux.params(est_params)

f(x, α, β, δ, γ) = hcat(
    α .* x[:, 1] .- β .* x[:, 1] .* x[:, 2],
    δ .* x[:, 1] .* x[:, 2] .- γ .* x[:, 2]
)

# maybe the arg expansion somehow forces grad to return tuple?
f(t, x) = f(x, est_params...)

opt = Descent(0.0001)
predict(t, x) = rk4(f, t, x)


for i in 1:800
    for (x₀, y, T) in train_loader
        t = hcat(LinRange.(0, T, 100)...)
        grad = gradient(θ) do 
            loss = Flux.Losses.mse(predict(t, x₀), y)
            return loss
        end
        for p in θ
            # TODO figure out why grad[est_params] is a tuple
            update!(opt, p, [grad[p]...])
        end
    end
end

println(est_params)

p1 = plot(
    lkv_train_samples.t,
    transpose(lkv_train_samples),
    title="LKV Model",
    label=["x₁ train" "x₂ train"],
    seriestype=:scatter
)
plot!(
    p1,
    lkv_test_samples.t,
    transpose(lkv_test_samples),
    label=["actual x₁" "actual x₂"]
)
lkv_est_samples = sample_lotka_volterra(lkv_ic, 500, 100, est_params...)
plot!(
    p1,
    lkv_est_samples.t,
    transpose(lkv_est_samples),
    label=["estimated x₁" "estimated x₂"]
)

## Train rk4 with fully connected network
fc = Chain(
    Dense(2, 16, relu),
    Dense(16, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 2)
)
θ = params(fc)
f(t, x) = fc(x')'

opt = Descent(0.0001)
predict(t, x) = rk4(f, t, x)

# TODO get CUDA working
for i in 1:10
    for (x₀, y, T) in train_loader
        t = hcat(LinRange.(0, T, 100)...)
        grad = gradient(θ) do 
            loss = Flux.Losses.mse(predict(t, x₀), y)
            return loss
        end
        # TODO this time it's all matrix/arrays...
        update!(opt, θ, grad)
    end
end

p1 = plot(
    lkv_train_samples.t,
    transpose(lkv_train_samples),
    title="LKV Model",
    label=["x₁ train" "x₂ train"],
    seriestype=:scatter
)
plot!(
    p1,
    lkv_test_samples.t,
    transpose(lkv_test_samples),
    label=["actual x₁" "actual x₂"]
)

function nn_lotka_volterra!(dx, x, p, t)
    dx[1], dx[2] = fc(x)
end

function sample_nn_lotka_volterra(x₀, n, T)
    sample_points = LinRange(0, T, n)
    tspan = (0.0, T)
    prob = ODEProblem(nn_lotka_volterra!, x₀, tspan, undef)
    return solve(prob, saveat=sample_points)
end

lkv_est_samples = sample_nn_lotka_volterra(lkv_ic, 500, 100)
plot!(
    p1,
    lkv_est_samples.t,
    transpose(lkv_est_samples),
    label=["estimated x₁" "estimated x₂"]
)