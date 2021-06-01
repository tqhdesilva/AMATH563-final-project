using DifferentialEquations


function sample_interval(start, stop, n)
    result = Vector{Float64}(undef, n)
    hm = Set{Float64}()
    for i in 1:n
        while true
            val = start + rand() * (stop - start)
            (val in hm) && continue
            result[i] = val
            push!(hm, val)
            break
        end
    end
    return result
end


function lotka_volterra!(dx, x, p, t)
    α, β, δ, γ = p
    dx[1] = α * x[1] - β * x[1] * x[2]
    dx[2] = δ * x[1] * x[2] - γ * x[2]
end


function sample_lotka_volterra(x₀, n, T, α, β, δ, γ; train=false)
    if train
        sample_points = sample_interval(0, T, n)
    else
        sample_points = LinRange(0, T, n)
    end
    tspan = (0.0, T)
    p = [α, β, δ, γ]
    prob = ODEProblem(lotka_volterra!, x₀, tspan, p)
    return solve(prob, saveat=sample_points)
end

