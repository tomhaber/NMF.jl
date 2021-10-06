# Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for large scale nonnegative matrix and tensor factorizations.",
# IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721, 2009.

function updateHALS!(grad::AbstractVector{T}, WT::AbstractMatrix{T}, XT::AbstractMatrix{S}, H::AbstractMatrix{T}, HTH::AbstractMatrix{T}, alpha::L1L2{T}) where {T, S}
    n, k = size(WT)
    norm = zero(T)

    if alpha.l2 > 0.0
        @inbounds for i in 1:k
            HTH[i,i] += alpha.l2
        end
    end

    @inbounds for j in 1:k
        hess = max(1e-10, HTH[j,j])

#		grad = XT*HT[:,j] - W * HHT[:,j]
        mul!(grad, XT, H[:,j])
        grad .-= alpha.l1
        mul!(grad, WT, view(HTH,:,j), -1.0, 1.0)

        w = @view WT[:,j]
        norm += projected_grad_norm(w, grad)
        axpy!(1/hess, grad, w)
        clamp_zero!(w)
    end

    norm
end

function updateW!(::L2NMF, grad::AbstractVector{T}, WT::AbstractMatrix{T}, XT::AbstractMatrix{S}, H::AbstractMatrix{T}, HTH::AbstractMatrix{T}) where {T,S}
    updateHALS!(grad, WT, XT, H, HTH, zero(L1L2{T}))
end

function updateW!(meas::RegularizedNMF{L2NMF}, grad::AbstractVector{T}, WT::AbstractMatrix{T}, XT::AbstractMatrix{S}, H::AbstractMatrix{T}, HTH::AbstractMatrix{T}) where {T,S}
    updateHALS!(grad, WT, XT, H, HTH, meas.alphaW)
end

function updateH!(::L2NMF, grad::AbstractVector{T}, H::AbstractMatrix{T}, X::AbstractMatrix{S}, WT::AbstractMatrix{T}, WWT::AbstractMatrix{T}) where {T,S}
    updateHALS!(grad, H, X, WT, WWT, zero(L1L2{T}))
end

function updateH!(meas::RegularizedNMF{L2NMF}, grad::AbstractVector{T}, H::AbstractMatrix{T}, X::AbstractMatrix{S}, WT::AbstractMatrix{T}, WWT::AbstractMatrix{T}) where {T,S}
    updateHALS!(grad, H, X, WT, WWT, meas.alphaH)
end

# WT[:,j] - (XT * H[:,j] - WT * HTH[:,]) / HTH[j,j]
# H[:,j] - (X * WT[:,j] - H * WWT[:,j]) / WWT[j,j]

function nmf!(meas::Union{RegularizedNMF{L2NMF}, L2NMF}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}, X::AbstractMatrix{S};
        atol::Real=1e-6, rtol::Real=1e-4, maxiter::Int=200, verbose::Bool=false) where {T, S}
    m, n = size(X)
    @assert size(HT,2) == m
    @assert size(W,2) == n
    k = size(HT,1)
    @assert size(W,1) == k

    HTH = WWT = Matrix{T}(undef, k ,k)
    gH = Vector{T}(undef, m)
    gW = Vector{T}(undef, n)

    WT = transpose(W)
    H = transpose(HT)
    XT = transpose(X)

    prev_obj = objective(meas, X, HT, W)

    iter = 1
    converged = false
    while ! converged && iter < maxiter
        # updateH
        mul!(WWT, W, WT)
        updateH!(meas, gH, H, X, WT, WWT)

        # updateW
        mul!(HTH, HT, H)
        updateW!(meas, gW, WT, XT, H, HTH)

        obj = objective(meas, X, HT, W)
        converged = (abs(obj - prev_obj) < rtol * abs(prev_obj)) || (abs(obj - prev_obj) < atol)
        verbose && println("objective changed from $prev_obj to $obj")
        prev_obj = obj
        iter += 1
    end

    converged
end

prefer_rowmajor(::Type{L2NMF}) = false
