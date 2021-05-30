using CUDA, Flux, Plots

include("rk4.jl")
include("lkv.jl")

x₀ = [0.8, 0.4]
sol = sample_lotka_volterra(x₀, 50, 100., 1.1, 0.4, 0.1, 0.4)
plot(sol)

# sample n times with random IC
# each sample has m points
# MSE loss at all points or just at end of sequence?
# irregularly sample?