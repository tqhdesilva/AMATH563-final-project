##
using Flux, Plots, Statistics, DiffEqFlux
using Flux.Optimise:update!
using Flux.Data:DataLoader
using ProgressMeter:Progress, next!
using DiffEqFlux:group_ranges

include("lkv.jl")
include("latent_ode.jl")

##
datasize = 500
lkv_ic = [0.8, 0.4]
true_params = [1.1, 0.4, 0.1, 0.4]
lkv_train_sol = sample_lotka_volterra(lkv_ic, datasize, 100, true_params...; train=true)
lkv_test_sol = sample_lotka_volterra(lkv_ic, datasize, 100, true_params...)

lkv_train_data = Array(lkv_train_sol)
tsteps = lkv_train_sol.t

# ##
# seqlen = 20
# train_loader = DataLoader(
#     (
#         [
#             lkv_train_sol[:, i:i - 1 + seqlen]
#             for i in 1:length(lkv_train_sol) + 1 - seqlen
#         ],
#         [
#             lkv_train_sol.t[i:i - 1 + seqlen]
#             for i in 1:length(lkv_train_sol) + 1 - seqlen
#         ]
#     ),
#     batchsize=8
# )
##
dzdt = FastChain(
    FastDense(2, 64, tanh),
    FastDense(64, 2)
)
p_init = initial_params(dzdt)

opt = ADAM(0.01)
ode_problem = ODEProblem((u, p, t) -> dzdt(u, p), [0.0, 0.0], (0.0, 100.0), p_init)
groupsize = 5
continuity_term = 200

anim = Animation()

function plot_multiple_shoot(plt, preds, group_size)
	step = group_size - 1
	ranges = group_ranges(datasize, group_size)

	for (i, rg) in enumerate(ranges)
		plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label=nothing)
	end
end

callback = function (p, l, preds; doplot=true)
    display(l)
    if doplot
        # plot the original data
        plt = scatter(tsteps, lkv_train_data[1,:], label="Data")

        # plot the different predictions for individual shoot
        plot_multiple_shoot(plt, preds, groupsize)

        frame(anim)
        display(plot(plt))
    end
    return false
end

function loss_function(data, pred)
	return sum(abs2, data - pred)
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, lkv_train_data, tsteps, ode_problem, loss_function, Tsit5(),
                          groupsize; continuity_term)
end

res_ms = DiffEqFlux.sciml_train(loss_multiple_shooting, p_init, opt; cb=callback, maxiters=300)
res_ms = DiffEqFlux.sciml_train(loss_multiple_shooting, res_ms.minimizer,
                                BFGS(), cb=callback, maxiters=100,
                                allow_f_increases=true)

gif(anim, "output/multiple_shooting.gif", fps=15)

##
est_solution = solve(ode_problem; u₀=lkv_test_sol[:, 30], saveat=lkv_test_sol.t, p=res_ms.minimizer)
plt = plot(est_solution.t, est_solution', labels=["estimated x₁" "estimated x₂"])
plot!(lkv_test_sol.t, lkv_test_sol', labels=["actual x₁" "actual x₂"])