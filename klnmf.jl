
function objective(X::AbstractMatrix, HT::AbstractMatrix, W::AbstractMatrix)
	WH = HT'*W

	obj = 0.0
	@inbounds for i in eachindex(X)
		obj += X[i]*log(X[i]+1e-5) - X[i]
		obj -= X[i]*log(WH[i]+1e-15) - WH[i]
	end

	obj
end

# g = sum(HT[:,q]) .- (X ./ (WH .+ 1e-10)) * HT[:,q]
# h = ((X ./ (WH .+ 1e-10).^2)) * HT[:,q].^2

# g = sum(W[:,q]) .- (X' ./ (WH' .+ 1e-10)) * W[:,q]
# h = ((X' ./ (WH' .+ 1e-10).^2)) * W[:,q].^2


			obj += delta
		end
	end
	obj
end

function updateH!(gh::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
	m,n = size(X)
	k = size(HT,1)

	nzv = nonzeros(X)
	rv = rowvals(X)

	fill!(gh, zero(T))
	gh[:,1] .= sum(W[q,:])

	@inbounds for j = 1:n
		xj = W[q,j]
		for idx in nzrange(X,j)
			v, i = nzv[idx], rv[idx]
			WHij = dot(view(HT,:,i), view(W,:,j)) + eps()

			gh[i, 1] -= (v / WHij) * xj
			gh[i, 2] += (v / WHij^2) * xj^2
		end
	end

	converged = true
	@inbounds for i = 1:m
		delta_i = -gh[i,1] / gh[i,2]
		newH = clamp(HT[q,i] + delta_i, eps(), Inf)
		converged &= (abs(newH - HT[q,i]) < abs(HT[q,i])*0.5)
		HT[q,i] = newH
	end

	converged
end

function updateW!(q::Int, X::SparseMatrixCSC{S}, HT::AbstractMatrix{T}, W::AbstractMatrix{T}) where {T,S}
	m,n = size(X)
	k = size(W,2)

	nzv = nonzeros(X)
	rv = rowvals(X)

	s = sum(HT[q,:])
	converged = true
	@inbounds for j = 1:n
		tmp_g = zero(T)
		tmp_h = zero(T)
		for idx in nzrange(X,j)
			v, i = nzv[idx], rv[idx]
			WHij = dot(view(HT,:,i), view(W,:,j)) + eps()

			tmp_g += (v / WHij) * HT[q,i]
			tmp_h += (v / WHij^2) * HT[q,i]^2
		end

		delta_j = (tmp_g - s) / tmp_h
		newW = clamp(W[q,j] + delta_j, eps(), Inf)
		converged &= (abs(newW - W[q,j]) < abs(W[q,j])*0.5)
		W[q,j] = newW
	end

	converged
end

function klnmf!(HT::AbstractMatrix{T}, W::AbstractMatrix{T}, X::AbstractMatrix{S}; tol=1e-4, maxiter=200, maxinner=2) where {T,S}
	m, n = size(X)
	k = size(HT,1)
	gh = Matrix{T}(undef, m, 2)

	iter = 1
	converged = false
	while ! converged && iter < maxiter
		@inbounds for q in 1:k
			for inner in 1:maxinner
				updateH!(gh, q, X, HT, W) && break
			end
		end

		@inbounds for q in 1:k
			for inner in 1:maxinner
				updateW!(q, X, HT, W) && break
			end
		end

		converged = false
		iter += 1
	end

	converged
end

function klnmf(X::AbstractMatrix{T}, k::Int; tol=1e-4, maxiter=200) where T
	m, n = size(X)
	avg = sqrt(mean(X) / k)
	HT = abs.(avg * randn(Float64, k, m))
	W = abs.(avg * randn(Float64, k, n))

	converged = klnmf!(HT, W, X; tol=tol, maxiter=maxiter)

	converged || @warn "failed to converge in $maxiter iterations"
	HT, W
end
