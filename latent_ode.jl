using DifferentialEquations, Flux, DiffEqFlux
using Flux.Losses:logitbinarycrossentropy


struct Encoder
    zdim::Int
    xdim::Int
    hdim::Int
    rnn_network::Chain
    function Encoder(zdim::Int, xdim::Int; hdim::Int=32)
        rnn_network = Chain(RNN(xdim, hdim), Dense(hdim, zdim * 2))
        new(zdim, xdim, hdim, rnn_network)
    end
end

function (e::Encoder)(x::Matrix{Float32})
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
    zdim::Int
    xdim::Int
    network::Chain
    function Decoder(zdim::Int, xdim::Int)
        network = Chain(
            Dense(zdim, 32, relu),
            Dense(32, 64, relu),
            Dense(64, 64, relu),
            Dense(64, xdim)
        )
        new(zdim, xdim, network)
    end
end

function (d::Decoder)(z::Union{Vector{Float32},Matrix{Float32}})
    return d.network(z)
end

struct VAE
    xdim::Int
    zdim::Int
    hdim::Int
    encoder::Encoder
    decoder::Decoder
    function VAE(xdim::Int, zdim::Int, hdim::Int)
        encoder = Encoder(zdim, xdim; hdim=hdim)
        decoder = Decoder(zdim, xdim)
        new(xdim, zdim, hdim, encoder, decoder)
    end
end

struct LatentODE
    dzdt::Chain
    vae::VAE
    β::AbstractFloat
    function LatentODE(xdim::Int, zdim::Int, hdim::Int, β::AbstractFloat=1.0)
        dzdt = Chain(
            Dense(zdim, 32, relu),
            Dense(32, 32, relu),
            Dense(32, zdim)
        )
        vae = VAE(xdim, zdim, hdim)
        new(dzdt, vae, β)
    end
end


function (l::LatentODE)(
    x::Matrix{Float32},
    t::Vector{Float32},
    tspan::Tuple{Number,Number}
)
    tspan = (max(tspan[1], t[1]), max(tspan[2], t[end]))
    μ, logσ = l.vae.encoder(x)
    z₀ = μ .+ randn(Float32, size(μ)...) .* exp.(logσ)
    node = NeuralODE(l.dzdt, tspan)
    sol = node(z₀)
    lode_sol(t) = l.vae.decoder(convert(Vector{Float32}, sol(t)))
    return lode_sol
end

function elbo(model::LatentODE, x::Array{Float32,3}, t::Matrix{Float32})
    batchsize = size(x)[end]
    tspans = [(t[1, i], t[end, i]) for i in 1:size(x)[end]]
    μ, logσ = model.vae.encoder(x)
    z₀ = μ .+ randn(Float32, size(μ)...) .* exp.(logσ)
    probs = [NeuralODE(model.dzdt, tspan; saveat=tᵢ) for (tspan, tᵢ) in zip(tspans, Flux.unstack(t, 2))]
    sols = [prob(z) for (prob, z) in zip(probs, Flux.unstack(z₀, 2))]
    x̂s = Flux.stack([model.vae.decoder(Flux.stack(sol.u, 2)) for sol in sols], 3)
    logp_x_z = - sum(logitbinarycrossentropy.(x̂s, x)) / batchsize # expectation
    kl_q_p = 0.5f0 * sum(@. (exp(logσ) + μ^2 - logσ - 1f0)) / batchsize
    elbo = logp_x_z - model.β * kl_q_p
    # TODO maybe add regularization (L2 reg) later
    return -elbo
end
