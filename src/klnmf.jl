import SparseArrays: SparseMatrixCSC, getcolptr, nonzeros, rowvals, nzrange

function reg_objective(HT::AbstractMatrix{T}, W::AbstractMatrix{T}, alphaH::L1L2{T}, alphaW::L1L2{T}) where T
	return alphaH.l1 * norm(HT,1) + alphaH.l2 * norm(HT,2) +
		alphaW.l1 * norm(W,1) + alphaW.l2 * norm(W,2)
end

function objective(X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}, alphaH::L1L2{T}, alphaW::L1L2{T}) where {T,S}
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

	obj + reg_objective(HT, W, alphaH, alphaW)
end

function objective(X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}, alphaH::L1L2{T}, alphaW::L1L2{T}, rs::Matrix{T}) where {T,S}
	m,n = size(X)

	nzv = nonzeros(X)
	rv = rowvals(X)

	obj = dot(view(rs, :, 1), view(rs, :, 2))

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
				v*log(v/(WHij + eps())) - v
			else
				WHij
			end
		end
	end

	obj + reg_objective(HT, W, alphaH, alphaW)
end

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

function klnmf!(HT::AbstractMatrix{T}, W::AbstractMatrix{T}, X::AbstractMatrix{S};
		tol=1e-4, maxiter=200, maxinner=2, alphaH::L1L2{T}=zero(L1L2{T}), alphaW::L1L2{T}=zero(L1L2{T})) where {T,S}
	m, n = size(X)
	k = size(HT,1)
	gh = Matrix{T}(undef, m, 2)
	rs = calc_rowsums(HT, W)

	prev_obj = objective(X, HT, W, alphaH, alphaW)

	iter = 1
	converged = false
	while ! converged && iter < maxiter
		@inbounds for q in 1:k
			for inner in 1:maxinner
				updateH!(gh, rs, q, X, HT, W, alphaH) && break
			end
		end

		@inbounds for q in 1:k
			for inner in 1:maxinner
				updateW!(rs, q, X, HT, W, alphaW) && break
			end
		end

		obj = objective(X, HT, W, alphaH, alphaW)
		converged = abs(obj - prev_obj) < tol * abs(prev_obj)
		prev_obj = obj
		iter += 1
	end

	converged
end

function klnmf(X::AbstractMatrix{S}, k::Int;
		tol=1e-4, maxiter=200, maxinner=2, alphaH::L1L2{Float64}=zero(L1L2{Float64}), alphaW::L1L2{Float64}=zero(L1L2{Float64})) where {S}
	m, n = size(X)
	avg = sqrt(mean(X) / k)
	HT = abs.(avg * randn(Float64, k, m))
	W = abs.(avg * randn(Float64, k, n))

	converged = klnmf!(HT, W, X; tol=tol, maxiter=maxiter, maxinner=maxinner, alphaH=alphaH, alphaW=alphaW)

	converged || @warn "failed to converge in $maxiter iterations"
	HT, W
end
