using DifferentialEquations


function lotka_volterra!(dx, x, p, t)
    α, β, δ, γ = p
    dx[1] = α * x[1] - β * x[1] * x[2]
    dx[2] = δ * x[1] * x[2] - γ * x[2]
end


function sample_lotka_volterra(x₀, n, T, α, β, δ, γ)
    sample_points = rand(Float64, n) * T
    tspan = (0.0, T)
    p = [α, β, δ, γ]
    prob = ODEProblem(lotka_volterra!, x₀, tspan, p)
    return solve(prob, saveat=sample_points)
end

