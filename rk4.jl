function rk4(f, t, x₀)
    xᵢ = transpose(x₀) # n x 2
    tᵢ = t[1, :]
    for i in 2:size(t)[1]
        h = t[i, :] .- tᵢ # n
        tᵢ = t[i, :]
        k₁ = f(tᵢ, xᵢ) # 2 x n
        k₂ = f(tᵢ ./ 2, xᵢ .+ h ./ 2 .* k₁)
        k₃ = f(tᵢ .+ h ./ 2, xᵢ .+ h ./ 2 .* k₂)
        k₄ = f(tᵢ .+ h, xᵢ .+ h .* k₃)
        xᵢ = xᵢ .+ 1 ./ 6 .* h .* (k₁ .+ 2 .* k₂ .+ 2 .* k₃ .+ k₄)
    end
    return collect(transpose(xᵢ))
end