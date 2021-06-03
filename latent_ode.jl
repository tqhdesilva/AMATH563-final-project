using DifferentialEquations, Flux


# TODO
# - implement VAE with RNN encoder

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

function (e::Encoder)(x::Array{Float32, 3})
    # x is vector of tsteps x xdim x batch
    # length(x) is # timesteps
    x = Flux.unstack(x, 1)
    Flux.reset!(e.rnn_network)
    out = e.rnn_network.(x)[end] # 2 * zdim x batch
    μ = out[1:e.zdim, :]
    logσ = out[e.zdim + 1:end, :]
    z = μ .+ randn(Float32, size(μ)...) .* exp.(logσ)
    return z
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

function (d::Decoder)(z::Matrix{Float32})
    # z is shape zdim x batchsize
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
    f::Function
    vae::VAE
end

function(::LatentODE)(x::Array{Float32, 3}, args..; kwargs)
    # x is tsteps x xdim x batchsize
    # return ODE solution
end

function elbo()
end

using DifferentialEquations
solve