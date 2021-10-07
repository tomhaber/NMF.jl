function random_initialize(rng::AbstractRNG, HT::AbstractMatrix{T}, W::AbstractMatrix{T}, X::AbstractMatrix) where T
    k = size(HT,1)
    avg = convert(T, sqrt(mean(X) / k))

    randf!(rng, X, scale=1.0) = (X .= abs.(scale .* randn!(rng, X)); X)
    randf!(rng, HT, avg);
    randf!(rng, W, avg);

#	for i in 1:k
#		x = @view W[:,i]
#		x ./= sum(x)
#	end

    HT, W
end

function initialize(rng::AbstractRNG, meas::Measure, X::AbstractMatrix, k::Int; dtype::Type=Float64)
    m,n = size(X)
    HT, W = if prefer_rowmajor(meas)
        HT = Matrix{dtype}(undef, k, m)
        W = Matrix{dtype}(undef, k, n)
        HT, W
    else
        H = Matrix{dtype}(undef, m, k)
        WT = Matrix{dtype}(undef, n, k)
        transpose(H), transpose(WT)
    end

    random_initialize(rng, HT, W, X)
end

initialize(meas::Measure, X::AbstractMatrix, k::Int; dtype::Type=Float64) = initialize(default_rng(), meas, X, k; dtype=dtype)
