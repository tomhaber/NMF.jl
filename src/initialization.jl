function random_initialize(::Type{T}, X::AbstractMatrix, k::Int) where T
    m,n = size(X)

    avg = convert(T, sqrt(mean(X) / k))
    HT = abs.(avg * randn(T, k, m))
    W = abs.(avg * randn(T, k, n))

#	for i in 1:k
#		x = @view W[:,i]
#		x ./= sum(x)
#	end

    HT, W
end

function initialize(X::AbstractMatrix, k::Int; dtype::Type=Float64)
    random_initialize(dtype, X, k)
end
