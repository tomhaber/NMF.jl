function normalize!(HT::AbstractMatrix, W::AbstractMatrix, p::Real=2)
    w = if p == 2
        sum(x->^(x,2), W, dims=2)
    elseif p == 1
        sum(abs, W, dims=2)
    elseif p == Inf
        maximum(W, dims=2)
    else
        error("unimplemented norm $p")
    end

    W ./ w
    HT .* w
    HT, W
end

normalize(HT::AbstractMatrix, W::AbstractMatrix, p::Real=2) = normalize!(copy(HT), copy(W), p)
