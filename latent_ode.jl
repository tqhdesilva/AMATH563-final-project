using DifferentialEquations, Flux, DiffEqFlux, CUDA
using Flux.Losses:logitbinarycrossentropy
import Flux:gpu, cpu


struct Encoder
    zdim::Int
    rnn_network::Chain
end

function new_encoder(zdim::Int, xdim::Int; hdim::Int=32)
    rnn_network = Chain(RNN(xdim, hdim), Dense(hdim, zdim * 2))
    return Encoder(zdim, rnn_network)
end

function (e::Encoder)(x::AbstractArray{Float32,2})
    # x is matrix of xdim x tsteps
    x = reverse(Flux.unstack(x, 2))
    Flux.reset!(e.rnn_network)
    out = e.rnn_network.(x)[end] # 2 * zdim
    μ = out[1:e.zdim]
    logσ = out[e.zdim + 1:end]
    return μ, logσ
end


function (e::Encoder)(x::Array{Float32,3})
    # x is a array of xdim x tsteps x batchsize
    x = reverse(Flux.unstack(x, 2))
    Flux.reset!(e.rnn_network)
    out = e.rnn_network.(x)[end] # 2 * zdim
    μ = out[1:e.zdim, :]
    logσ = out[e.zdim + 1:end, :]
    return μ, logσ
end

struct Decoder
    network::Chain
end

function new_decoder(zdim::Int, xdim::Int)
    network = Chain(
        Dense(zdim, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 32, relu),
        Dense(32, xdim)
    )
    return Decoder(network)
end

function (d::Decoder)(z::Union{AbstractArray{Float32,1},AbstractArray{Float32,2}})
    return d.network(z)
end

struct VAE
    encoder::Encoder
    decoder::Decoder
end

function new_vae(xdim::Int, zdim::Int, hdim::Int)
    encoder = new_encoder(zdim, xdim; hdim=hdim)
    decoder = new_decoder(zdim, xdim)
    return VAE(encoder, decoder)
end

struct LatentODE
    dzdt::Chain
    vae::VAE
    β::AbstractFloat
end

function new_latent_ode(xdim::Int, zdim::Int, hdim::Int; β::AbstractFloat=1.0)::LatentODE
    dzdt = Chain(
        Dense(zdim, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 32, relu),
        Dense(32, zdim)
    )
    vae = new_vae(xdim, zdim, hdim)
    return LatentODE(dzdt, vae, β)
end

gpu(enc::Encoder) = Encoder(enc.zdim, enc.rnn_network |> gpu)
cpu(enc::Encoder) = Encoder(enc.zdim, enc.rnn_network |> cpu)
gpu(dec::Decoder) = Decoder(dec.network |> gpu)
cpu(dec::Decoder) = Decoder(dec.network |> cpu)
gpu(vae::VAE) = VAE(vae.encoder |> gpu, vae.decoder |> gpu)
cpu(vae::VAE) = VAE(vae.encoder |> cpu, vae.decoder |> cpu)
gpu(lode::LatentODE) = LatentODE(lode.dzdt |> gpu, lode.vae |> gpu, lode.β)
cpu(lode::LatentODE) = LatentODE(lode.dzdt |> cpu, lode.vae |> cpu, lode.β)

function (l::LatentODE)(
    x::AbstractArray{Float32,2},
    t::AbstractArray{Float32,1},
    tspan::Tuple{Number,Number}
)
    tspan = (max(tspan[1], t[1]), max(tspan[2], t[end]))
    μ, logσ = l.vae.encoder(x)
    noise = randn(Float32, size(μ)...)
    if isa(x, CUDA.CuArray)
        noise = noise |> gpu
    end
    z₀ = μ .+ noise .* exp.(logσ)
    node = NeuralODE(l.dzdt, tspan)
    sol = node(z₀)
    lode_sol(t) = l.vae.decoder(convert(Vector{Float32}, sol(t)))
    return lode_sol
end


function elbo(model::LatentODE, x::AbstractArray{Float32,2}, t::AbstractArray{Float32,1})
    tspan = (t[1], t[end])
    μ, logσ = model.vae.encoder(x)
    noise = randn(Float32, size(μ)...)
    if isa(x, CUDA.CuArray)
        noise = noise |> gpu
    end
    z₀ = μ .+ noise .* exp.(logσ) # TODO send this to gpu when necessary
    node = NeuralODE(model.dzdt, tspan; saveat=t)
    sol = node(z₀)
    x̂ = model.vae.decoder(Flux.stack(sol.u, 2))
    logp_x_z = -sum(logitbinarycrossentropy.(x̂, x))
    kl_q_p = 0.5 * sum(exp.(logσ) .+ μ.^2 .- logσ .- 1)
    elbo = logp_x_z - model.β * kl_q_p
    return -elbo
end