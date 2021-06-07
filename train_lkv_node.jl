using Flux, Plots, Statistics
using Flux.Optimise:update!
using Flux.Data:DataLoader

include("lkv.jl")
include("latent_ode.jl")


lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_sol = sample_lotka_volterra(lkv_ic, 20, 5, true_params...; train=true)
lkv_test_sol = sample_lotka_volterra(lkv_ic, 500, 100, true_params...)

dzdt = Chain(Dense(2, 64, sigmoid), Dense(64, 2))
tspan = (0.0, 5.0)
z = Array(lkv_train_sol)
t = lkv_train_sol.t
z₀ = z[:, 1]
n_ode = NeuralODE(dzdt, tspan, saveat=t, reltol=1e-7, abstol=1e-9)
predict_n_ode() = n_ode(z₀)
loss_n_ode() = sum((z .- predict_n_ode()).^2)


data = Iterators.repeated((), 1000)
opt = ADAM(0.1)
cb = function () # callback function to observe training
    display(loss_n_ode())
    # plot current prediction against data
    cur_pred = predict_n_ode()
    pl = scatter(t, z[1,:], label="data")
    scatter!(pl, t, cur_pred[1,:], label="prediction")
    display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

ps = Flux.params(n_ode)
Flux.train!(loss_n_ode, ps, data, opt, cb=cb)