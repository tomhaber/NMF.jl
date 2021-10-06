function updateH!(gh::Matrix{T}, rs::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}, alphaH::L1L2{T}) where {T,S}
    m,n = size(X)
    k = size(HT,1)

    nzv = nonzeros(X)
    rv = rowvals(X)

    gh[:,1] .= rs[q, 2] + alphaH.l1*n
    gh[:,2] .= zero(T)

    @inbounds for j = 1:n
        xj = W[q,j]
        for idx in nzrange(X,j)
            v, i = nzv[idx], rv[idx]
            WHij = dot(view(HT,:,i), view(W,:,j)) + eps()

            gh[i, 1] -= (v / WHij) * xj
            gh[i, 2] += (v / WHij^2) * xj^2
        end
    end

    rowsum_Hq = zero(T)
    converged = true
    @inbounds for i = 1:m
        delta_i = -(gh[i,1] + 2n*alphaH.l2*HT[q,i]) / (gh[i,2] + 2n*alphaH.l2)
        newH = clamp(HT[q,i] + delta_i, eps(), Inf)
        converged &= (abs(newH - HT[q,i]) < abs(HT[q,i])*0.5)
        HT[q,i] = newH
        rowsum_Hq += newH
    end

    rs[q, 1] = rowsum_Hq
    converged
end

function updateH!(::KLNMF, gh::Matrix{T}, rs::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
    updateH!(gh, rs, q, X, HT, W, zero(L1L2{T}))
end

function updateH!(m::RegularizedNMF{KLNMF}, gh::Matrix{T}, rs::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
    updateH!(gh, rs, q, X, HT, W, m.alphaH)
end

function updateW!(rs::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}, alphaW::L1L2{T}) where {T,S}
    m,n = size(X)
    k = size(W,2)

    nzv = nonzeros(X)
    rv = rowvals(X)

    rowsum_Wq = zero(T)
    s = rs[q, 1] + alphaW.l1*m

    converged = true
    @inbounds for j = 1:n
        tmp_g = zero(T)
        tmp_h = zero(T)
        for idx in nzrange(X,j)
            v, i = nzv[idx], rv[idx]
            WHij = dot(view(HT,:,i), view(W,:,j)) + eps()

            tmp_g -= (v / WHij) * HT[q,i]
            tmp_h += (v / WHij^2) * HT[q,i]^2
        end

        delta_j = -(tmp_g + s + 2m*alphaW.l2*W[q,j]) / (tmp_h + 2m*alphaW.l2)
        newW = clamp(W[q,j] + delta_j, eps(), Inf)
        converged &= (abs(newW - W[q,j]) < abs(W[q,j])*0.5)
        W[q,j] = newW
        rowsum_Wq += newW
    end

    rs[q, 2] = rowsum_Wq
    converged
end

function updateW!(::KLNMF, rs::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
    updateW!(rs, q, X, HT, W, zero(L1L2{T}))
end

function updateW!(m::RegularizedNMF{KLNMF}, rs::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
    updateW!(rs, q, X, HT, W, m.alphaW)
end

function nmf!(meas::Union{KLNMF, RegularizedNMF{KLNMF}},
        HT::AbstractMatrix{T}, W::AbstractMatrix{T}, X::AbstractMatrix{S};
        atol=1e-6, rtol=1e-4, maxiter=200, maxinner=2, verbose=true) where {T,S}
    m, n = size(X)
    @assert size(HT,2) == m
    @assert size(W,2) == n
    k = size(HT,1)
    @assert size(W,1) == k

    gh = Matrix{T}(undef, m, 2)
    rs = calc_rowsums(HT, W)

    prev_obj = objective(meas, X, HT, W, rs)

    iter = 1
    converged = false
    while ! converged && iter < maxiter
        @inbounds for q in 1:k
            for inner in 1:maxinner
                updateH!(meas, gh, rs, q, X, HT, W) && break
            end
        end

        @inbounds for q in 1:k
            for inner in 1:maxinner
                updateW!(meas, rs, q, X, HT, W) && break
            end
        end

        obj = objective(meas, X, HT, W, rs)
        converged = (abs(obj - prev_obj) < rtol * abs(prev_obj)) || (abs(obj - prev_obj) < atol)
        verbose && println("iteration $iter: objective changed from $prev_obj to $obj ($(abs(obj - prev_obj)/abs(prev_obj)))")
        prev_obj = obj
        iter += 1
    end

    converged
end

prefer_rowmajor(::Type{KLNMF}) = true
