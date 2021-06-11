## Create datasets, loaders
using Flux, Plots, Statistics
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

## Train rk4 model to learn lkv params
est_params = rand(4)
println(est_params)
θ = Flux.params(est_params)

function f′(x, p)
    α, β, δ, γ = p
    return hcat(
    α .* x[:, 1] .- β .* x[:, 1] .* x[:, 2],
    δ .* x[:, 1] .* x[:, 2] .- γ .* x[:, 2]
)
end

# Varargs does not work when currying
# Grad returns a tuple
f(t, x) = f′(x, est_params)

opt = Descent(0.0001)
predict(t, x) = rk4(f, t, x)


for i in 1:800
    for (x₀, y, T) in train_loader
        t = hcat(LinRange.(0, T, 100)...)
        grad = gradient(θ) do 
            loss = Flux.Losses.mse(predict(t, x₀), y)
            return loss
        end
        update!(opt, θ, grad)
    end
end

println(est_params)

p1 = plot(
    lkv_train_samples.t,
    transpose(lkv_train_samples),
    title="LKV Model using LKV equation",
    label=["x₁ train" "x₂ train"],
    seriestype=:scatter
)
plot!(
    p1,
    lkv_test_samples.t,
    transpose(lkv_test_samples),
    label=["actual x₁" "actual x₂"],
    linewidth=2
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
    Dense(2, 32, relu),
    Dense(32, 64, relu),
    Dense(64, 16, relu),
    Dense(16, 2)
)
θ = params(fc)
f(t, x) = fc(x')'

opt = ADAMW(0.0001)
predict(t, x) = rk4(f, t, x)

function train!(loader)
    Flux.testmode!(fc, false)
    for (x₀, y, T) in loader
        t = hcat(LinRange.(0, T, 100)...)
        grad = gradient(θ) do 
            loss = Flux.Losses.mse(predict(t, x₀), y)
            return loss
        end
        # TODO this time it's all matrix/arrays...
        update!(opt, θ, grad)
    end
end
@Flux.epochs 800 train!(train_loader)


p1 = plot(
    lkv_train_samples.t,
    transpose(lkv_train_samples),
    title="LKV Model using NN",
    label=["x₁ train" "x₂ train"],
    seriestype=:scatter
)
plot!(
    p1,
    lkv_test_samples.t,
    transpose(lkv_test_samples),
    label=["actual x₁" "actual x₂"],
    linewidth=2
)

function nn_lotka_volterra!(dx, x, p, t)
    dx[1], dx[2] = fc(x)
end

function sample_nn_lotka_volterra(x₀, n, T)
    Flux.testmode!(fc, true)
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
