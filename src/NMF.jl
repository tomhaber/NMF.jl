module NMF
    import Statistics: mean
    import Random: AbstractRNG, default_rng, rand!, rand, randn!
    import LinearAlgebra: axpby!, axpy!, mul!, dot, norm
    import SparseArrays: SparseMatrixCSC, getcolptr, nonzeros, rowvals, nzrange

    struct L1L2{T <: Real}
        l1::T
        l2::T
    end

    Base.zero(::Type{L1L2{T}}) where {T <: Real} = L1L2{T}(zero(T), zero(T))
    Base.collect(x::L1L2) = (x.l1, x.l2)

    include("utils.jl")
    include("measures.jl")
    include("initialization.jl")
    include("normalize.jl")
    include("nmf.jl")
    include("klnmf.jl")

    prefer_rowmajor(meas::Measure) = prefer_rowmajor(typeof(meas))
    prefer_rowmajor(::Type{<:Measure}) = false # fallback
    prefer_rowmajor(::Type{RegularizedNMF{M}}) where M = prefer_rowmajor(M)

    function nmf(rng::AbstractRNG, meas::Measure, X::AbstractMatrix, k::Int; dtype::Type=Float64, maxiter=200, kw...)
        HT, W = initialize(rng, meas, X, k, dtype=dtype)
        converged = nmf!(meas, HT, W, X; maxiter=maxiter, kw...)

        converged || @warn "failed to converge in $maxiter iterations"
        HT, W
    end

    nmf(meas::Measure, X::AbstractMatrix, k::Int; kw...) = nmf(default_rng(), meas, X, k; kw...)

    export initialize, objective
    export normalize, normalize!
    export nmf, nmf!
    export L1L2, L2NMF, KLNMF, RegularizedNMF
end
