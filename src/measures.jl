abstract type Measure end

"""
    l_2 NMF: Gaussian noise
"""
struct L2NMF <: Measure
end

"""
    KL NMF: Poisson noise
"""
struct KLNMF <: Measure
end

struct RegularizedNMF{M <: Measure, T} <: Measure
    measure::M
    alphaW::L1L2{T}
    alphaH::L1L2{T}
end

function RegularizedNMF(measure::M; alphaW::L1L2{T}=zero(L1L2{T}), alphaH::L1L2{T}=zero(L1L2{T})) where {M <: Measure, T}
    RegularizedNMF{M,T}(measure, alphaW, alphaH)
end

function reg_objective(W::AbstractMatrix{T}, alpha::L1L2{T}) where T
    return alpha.l1 * norm(W,1) + alpha.l2 * norm(W,2)
end

function objective(reg::RegularizedNMF, X::AbstractMatrix{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T, S}
    objective(reg.measure, X, HT, W) + reg_objective(HT, reg.alphaH) + reg_objective(W, reg.alphaW)
end

function objective(::L2NMF, X::AbstractMatrix{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T, S}
    norm(X .- HT'*W, 2)^2
end

function objective(::L2NMF, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T, S}
    m,n = size(X)

    nzv = nonzeros(X)
    rv = rowvals(X)

    obj = zero(T)
    idx = getcolptr(X)[1]
    @inbounds for j in 1:n
        next = getcolptr(X)[j+1]

        for i in 1:m
            HTi = view(HT, :, i)
            Wj = view(W, :, j)
            WHij = dot(HTi, Wj)

            obj += if idx < next && rv[idx] == i
                v = nzv[idx]
                idx += 1
                (v - WHij)^2
            else
                WHij^2
            end
        end
    end
    obj
end

function objective(::KLNMF, X::AbstractMatrix{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T, S}
    WH = HT'*W
    mask = X .> 0
    sum(@. X[mask] * log(X[mask] / (WH[mask] + eps()))) - sum(X) + sum(WH)
end

function objective(::KLNMF, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T, S}
    m,n = size(X)

    nzv = nonzeros(X)
    rv = rowvals(X)

    obj = zero(T)
    idx = getcolptr(X)[1]
    @inbounds for j in 1:n
        next = getcolptr(X)[j+1]

        for i in 1:m
            HTi = view(HT, :, i)
            Wj = view(W, :, j)
            WHij = dot(HTi, Wj)

            obj += if idx < next && rv[idx] == i
                v = nzv[idx]
                idx += 1
                v*log(v/(WHij + eps())) - v + WHij
            else
                WHij
            end
        end
    end
    obj
end
