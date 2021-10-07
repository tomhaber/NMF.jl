function calc_rowsums!(rs::Matrix{T}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where T
    sum!(view(rs, :, 1), HT)
    sum!(view(rs, :, 2), W)
    rs
end

function calc_rowsums(HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where T
    k = size(HT, 1)
    @assert size(W,1) == k

    rs = Matrix{T}(undef, (k, 2))
    calc_rowsums!(rs, HT, W)
end

@inline function assert_allequal_size(X)
    n = size(X[1], 2)
    @inbounds for i in 2:length(X)
        @assert size(X[i], 2) == n "not all matrices have equal size ($(size(X[i], 2)) <=> $n)"
    end
    n
end

function projected_grad_norm(w::AbstractVector{T}, g::AbstractVector{T}) where T
    norm = zero(T)

    @inbounds for i = 1:length(w)
        pg = if iszero(w[i])
            min(zero(T), g[i])
        else
            g[i]
        end

        norm += pg^2
    end

    norm
end

function clamp_zero!(w::AbstractVector{T}) where T
    @inbounds for i = 1:length(w)
        w[i] = max(w[i], zero(T))
    end
end

function check_zeroinflation(X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
    m,n = size(X)

    rv = rowvals(X)
    nzeros = length(X) - nnz(X)

    pred_zero = zero(T)
    @inbounds for j in 1:n
        for i in 1:m
            mu = dot(view(HT,:,i), view(W,:,j))
            pred_zero += pdf(Poisson(mu), 0)
        end
    end

    nzeros / pred_zero
end

function check_overdispersion(X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
    m,n = size(X)

    rv = rowvals(X)
    nzv = nonzeros(X)
    k = length(HT) + length(W)

    chisq = zero(T)
    idx = getcolptr(X)[1]
    @inbounds for j in 1:n
        next = getcolptr(X)[j+1]

        for i in 1:m
            mu = dot(view(HT,:,i), view(W,:,j))

            y = if idx < next && rv[idx] == i
                v = nzv[idx]
                idx += 1
                v
            else
                0
            end

            chisq += (y - mu)^2 / mu
        end
    end

    n = m*n
    n, k, chisq, chisq / (n - k), ccdf(Chisq(n - k), chisq)
end


function predict(HT::AbstractMatrix{T}, W::AbstractMatrix{T}, X::AbstractMatrix{S}) where {T,S}
  Y = similar(X, T)

    nzv = nonzeros(Y)
    rv = rowvals(X)

    @inbounds for j = 1:size(X,2)
        for idx in nzrange(X,j)
            i = rv[idx]
            nzv[idx] = dot(view(HT,:,i), view(W,:,j))
        end
    end

    Y
end
