import Statistics: mean
import Random: randn, seed!
import LinearAlgebra: diagind

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

function projected_grad(w::T, g::T) where T
	if iszero(w)
		min(zero(T), g)
	else
		g
	end
end

function updateHALS!(W::AbstractMatrix{T}, XHT::AbstractMatrix{T}, HHT::AbstractMatrix{T}, alpha::Float64) where T
	if alpha > 0.0
		@inbounds HHT[diagind(HHT)] .+= alpha
	end

	n, k = size(W)
	norm = zero(T)

	for j in 1:k
		@inbounds hess = max(1e-10, HHT[j,j])
		G = W * HHT[:,j] - XHT[:,j]
		for i in 1:n
			#@inbounds grad = dot(W[i,:], HHT[:,j]) - XHT[i,j]
			@inbounds grad = G[i]
			@inbounds norm += projected_grad(W[i,j], grad)^2
			@inbounds W[i,j] = max(W[i,j] - grad / hess, zero(T))
		end
	end

	norm
end

function nmf(X::AbstractMatrix{T}, k::Int; tol=1e-4, maxiter=200, alpha=0.0) where T
	seed!(1234)
	W, H = initialize_nmf(X, k)

	m, n = size(X)
	WTW = HHT = Matrix{T}(undef, k ,k)
	XHT = Matrix{T}(undef, m, k)
	XTW = Matrix{T}(undef, n, k)

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
		mul!(XHT, X, HT)
		norm += updateHALS!(W, XHT, HHT, alpha)

		# updateH
		mul!(WTW, WT, W)
		mul!(XTW, XT, W)
		norm += updateHALS!(HT, XTW, WTW, alpha)

		if iter == 1
			norm0 = norm
		end

		println("$(LinearAlgebra.norm(X .- W*H)) $norm $norm0")
		converged = norm / norm0 < tol
		iter += 1
	end

	converged || error("failed to converge in $maxiter iterations")
	W, H
end
