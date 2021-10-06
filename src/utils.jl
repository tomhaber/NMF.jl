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
