import Statistics: mean
import Random: randn, seed!
import LinearAlgebra: axpy!, mul!, dot, norm
import SparseArrays: AbstractSparseMatrix, nonzeros, rowvals, nzrange

# Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for large scale nonnegative matrix and tensor factorizations.",
# IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721, 2009.

function initialize_nmf(X::AbstractMatrix{T}, k::Int) where T
	m,n = size(X)

	avg = sqrt(mean(X) / k)
	W = abs.(avg * randn(T, m, k))
	H = abs.(avg * randn(T, k, n))

	for i in 1:k
		x = @view W[:,i]
		x ./= sum(x)
	end

	W, H
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

objective(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix) = norm(X .- W*H)
function objective(X::AbstractSparseMatrix{T}, W::AbstractMatrix{T}, H::AbstractMatrix{T}) where T
	norm = zero(T)

	for col in 1:size(X, 2)
		for i in nzrange(X, col)
			row = rowvals(X)[i]
			val = nonzeros(X)[i]
			norm += (val - dot(view(W,row,:), view(H,:,col)))^2
		end
	end

	sqrt(norm)
end

function updateHALS!(grad::AbstractVector{T}, W::AbstractMatrix{T}, X::AbstractMatrix{T}, HT::AbstractMatrix{T}, HHT::AbstractMatrix{T}, alpha::Tuple{Float64, Float64}) where T
	n, k = size(W)
	norm = zero(T)

	if alpha[2] > 0.0
		@inbounds for i in 1:k
			HHT[i,i] += alpha[2]
		end
	end

	@inbounds for j in 1:k
		hess = max(1e-10, HHT[j,j])

#		grad = X*HT[:,j] - W * HHT[:,j]
		mul!(grad, X, HT[:,j])
		grad .-= alpha[1]
		mul!(grad, W, view(HHT,:,j), -1.0, 1.0)

		w = @view W[:,j]
		norm += projected_grad_norm(w, grad)
		axpy!(1/hess, grad, w)
		clamp_zero!(w)
	end

	norm
end

# W[:,j] - (X*HT[:,j] - W * HHT[:,]) / HHT[j,j]
# HT[:,j] - (XT * W[:,j] - HT * WTW[:,j]) / WTW[j,j]

# H'V'VH

function nmf(X::AbstractMatrix{T}, k::Int; tol=1e-4, maxiter=200, alphaW::Tuple{Float64, Float64}=(0.0,0.0), alphaH::Tuple{Float64, Float64}=(0.0,0.0)) where T
	seed!(1234)
	W, H = initialize_nmf(X, k)

	m, n = size(X)
	WTW = HHT = Matrix{T}(undef, k ,k)
	gW = Vector{T}(undef, m)
	gH = Vector{T}(undef, n)

	HT = transpose(H)
	WT = transpose(W)
	XT = transpose(X)
	norm0 = zero(T)

	iter = 1
	converged = false
	while ! converged && iter < maxiter
		norm = zero(T)

		# updateW
		mul!(HHT, H, HT)
		norm += updateHALS!(gW, W, X, HT, HHT, alphaW)

		# updateH
		mul!(WTW, WT, W)
		norm += updateHALS!(gH, HT, XT, W, WTW, alphaH)

		if iter == 1
			norm0 = norm
		end

#println("$(LinearAlgebra.norm(X .- W*H)) $norm $norm0")
		converged = norm / norm0 < tol
		iter += 1
	end

	converged || error("failed to converge in $maxiter iterations")
	W, H
end
