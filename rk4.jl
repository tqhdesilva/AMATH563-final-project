function rk4(f, t, initial_value)
    tₙ, yₙ  = t[1], initial_value
    results = Vector(Float64, length(points))
    for (i, tₙ₊₁) in enumerate(t)
        h = tₙ₊₁ - tₙ
        k₁ = f(tₙ, yₙ)
        k₂ = f(tₙ + h / 2, yₙ + h / 2 * k₁)
        k₃ = f(tₙ + h / 2, yₙ + h / 2 * k₂)
        k₄ = f(tₙ + h, yₙ + h * k₃)
        yₙ += 1 / 6 * h * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
        tₙ = tₙ₊₁
        results[i] = yₙ
    end
end

