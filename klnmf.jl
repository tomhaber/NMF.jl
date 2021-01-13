import Statistics: mean
import Random: randn
import LinearAlgebra: mul!, dot

function objective(X::AbstractMatrix, W::AbstractMatrix, HT::AbstractMatrix)
	WH = W*HT'

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


#=
\begin{align*}
\partial H_{i,q}^{T} = \sum_j \left( W_{j,q} - \frac{X_{j,i} W_{j,q}}{\sum_k W_{j,k} H_{i,k}^{T}} \right)\\
\partial^2 H_{i,q}^{T} = \sum_j \frac{X_{j,i} W_{j,q}^2}{(\sum_k W_{j,k} H_{i,k}^{T})^2}
\end{align*}
=#

function updateW!(gh::Matrix{T}, q::Int, X::SparseMatrixCSC{S}, W::AbstractMatrix{T}, HT::AbstractMatrix{T}) where {T,S}
	m,n = size(X)
	k = size(W,2)

	nzv = nonzeros(X)
	rv = rowvals(X)

	fill!(gh, zero(T))
	gh[:,1] .= sum(HT[:,q])

	@inbounds for j = 1:n
		xj = HT[j,q]
		for idx in nzrange(X,j)
			v, i = nzv[idx], rv[idx]
			WHij = dot(W[i,:], HT[j,:]) + eps()

			gh[i, 1] -= (v / WHij) * xj
			gh[i, 2] += (v / WHij^2) * xj^2
		end
	end

	converged = true
	@inbounds for i = 1:m
		delta_i = -gh[i,1] / gh[i,2]
		newW = clamp(W[i,q] + delta_i, eps(), Inf)
		converged &= (abs(newW - W[i,q]) < abs(W[i,q])*0.5)
		W[i,q] = newW
	end

	converged
end

function updateH!(q::Int, X::SparseMatrixCSC{S}, W::AbstractMatrix{T}, HT::AbstractMatrix{T}) where {T,S}
	m,n = size(X)
	k = size(W,2)

	nzv = nonzeros(X)
	rv = rowvals(X)

	s = sum(W[:,q])
	converged = true
	@inbounds for j = 1:n
		tmp_g = zero(T)
		tmp_h = zero(T)
		for idx in nzrange(X,j)
			v, i = nzv[idx], rv[idx]
			WHij = dot(W[i,:], HT[j,:]) + eps()

			tmp_g += (v / WHij) * W[i,q]
			tmp_h += (v / WHij^2) * W[i,q]^2
		end

		delta_j = (tmp_g - s) / tmp_h
		newH = clamp(HT[j,q] + delta_j, eps(), Inf)
		converged &= (abs(newH - HT[j,q]) < abs(HT[j,q])*0.5)
		HT[j,q] = newH
	end

	converged
end

function klnmf!(W::AbstractMatrix{T}, HT::AbstractMatrix{T}, X::AbstractMatrix{S}; tol=1e-4, maxiter=200, maxinner=2) where {T,S}
	m, n = size(X)
	gh = Matrix{T}(undef, m, 2)

	iter = 1
	converged = false
	while ! converged && iter < maxiter
		# updateW
		@inbounds for q in 1:k
			for inner in 1:maxinner
				updateW!(gh, q, X, W, HT) && break
			end
		end

		# updateH
		@inbounds for q in 1:k
			for inner in 1:maxinner
				updateH!(q, X, W, HT) && break
			end
		end

		converged = false
		iter += 1
	end

	converged
end

function update!(Wt::AbstractVector{T}, WHt::AbstractVector{T}, Vt::AbstractVector{S}, HT::AbstractMatrix{T}) where {T, S}
	n, k = size(HT)

	@inbounds for q in 1:k
		for inner in 1:2
			g = 0
			h = 0
			for j in 1:n
				tmp = Vt[j]/(WHt[j] + 1e-10)
				g = g + HT[j, q] * (1-tmp)
				h = h + HT[j, q]*HT[j, q]*tmp/(WHt[j]+1e-10)
			end

			s = -g/h;
			oldW = Wt[q]
			newW = Wt[q] + s
			if newW < 1e-15
				newW = 1e-15
			end

			diff = newW - oldW
			Wt[q] = newW

			for j in 1:n
				WHt[j] = WHt[j] + diff * HT[j, q];
				if WHt[j] < 1e-16
					WHt[j] = 1e-16;
				end
			end

			(abs(diff) < abs(oldW)*0.5) && break;
		end
	end
end

function klnmf_old!(W::AbstractMatrix{T}, HT::AbstractMatrix{T}, X::AbstractMatrix{S}; tol=1e-4, maxiter=200) where {T,S}
	m, n = size(X)
	WH = Matrix{T}(undef, m ,n)
	WHT = Vector{T}(undef, m)
	VT = Vector{T}(undef, m)

	mul!(WH, W, HT')

	iter = 1
	converged = false
	while ! converged && iter < maxiter
		# updateW
		for i in 1:m
			Wt = view(W,i,:)
			WHt = view(WH,i,:)
			Vt = X[i,:]
			update!(Wt, WHt, Vt, HT)
		end

		# updateH
		for j in 1:n
			Ht = view(HT,j,:)
			WHt = view(WH,:,j)
			Vt = X[:,j]
			update!(Ht, WHt, Vt, W)
		end

		converged = false
		iter += 1
	end

	converged
end

function klnmf(X::AbstractMatrix{T}, k::Int; tol=1e-4, maxiter=200) where T
	m, n = size(X)
	avg = sqrt(mean(X) / k)
	W = abs.(avg * randn(Float64, m, k))
	HT = abs.(avg * randn(Float64, n, k))

	converged = klnmf!(W, HT, X; tol=tol, maxiter=maxiter)

	converged || @warn "failed to converge in $maxiter iterations"
	W, HT
end

function klnmf_orig(X::AbstractMatrix, k::Int; maxiter=200, maxtime=100.)
	m,n = size(X)
	V = convert(Matrix{Float64}, X)

	trace = 1
	obj = Vector{Float64}(undef, maxiter)
	time = Vector{Float64}(undef, maxiter)

	avg = sqrt(mean(X) / k)
	WT = abs.(avg * randn(Float64, k, m))
	H = abs.(avg * randn(Float64, k, n))

	len = ccall(("newKL", libcell), Cint,
		(Cint, Cint, Cint, Cint, Cdouble, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}),
		m, n, k, maxiter, maxtime, V, WT, H, trace, obj, time)

	WT, H, obj, time
end
