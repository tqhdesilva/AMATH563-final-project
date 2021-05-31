using CUDA, Flux, Plots, Statistics
using Flux.Optimise:update!
using Flux.Data:DataLoader


include("rk4.jl")
include("lkv.jl")

lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_samples = sample_lotka_volterra(lkv_ic, 200, 100., true_params...; train=true)
lkv_test_samples = sample_lotka_volterra(lkv_ic, 500, 100, true_params...)

lkv_train_tuple = (
    lkv_train_samples[:, 1:end - 1],
    lkv_train_samples[:, 2:end],
    lkv_train_samples.t[2:end] .- lkv_train_samples.t[1:end - 1]
)

train_loader = DataLoader(lkv_train_tuple; batchsize=16, shuffle=true)
test_loader = DataLoader(lkv_train_samples; batchsize=32, shuffle=false)

est_params = rand(4)
println(est_params)
θ = Flux.params(est_params)

f(x, α, β, δ, γ) = hcat(
    α .* x[:, 1] .- β .* x[:, 1] .* x[:, 2],
    δ .* x[:, 1] .* x[:, 2] .- γ .* x[:, 2]
)

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