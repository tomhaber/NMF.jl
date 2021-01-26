# Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for large scale nonnegative matrix and tensor factorizations.",
# IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721, 2009.

function initialize_nmf(::Type{T}, X::AbstractMatrix, k::Int) where T
	m,n = size(X)

	avg = sqrt(mean(X) / k)
	W = abs.(avg * randn(T, m, k))
	HT = abs.(avg * randn(T, n, k))

#	for i in 1:k
#		x = @view W[:,i]
#		x ./= sum(x)
#	end

	W, HT
end

#objective(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix) = norm(X .- W*H)

function updateHALS!(grad::AbstractVector{T}, W::AbstractMatrix{T}, X::AbstractMatrix{S}, HT::AbstractMatrix{T}, HHT::AbstractMatrix{T}, alpha::L1L2{T}) where {T, S}
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

function nmf!(W::Matrix{T}, HT::Matrix{T}, X::AbstractMatrix{S}, k::Int;
		 tol=1e-4, maxiter=200, alphaW::L1L2{T}=zero(L1L2{T}), alphaH::L1L2{T}=zero(L1L2{T})) where {T, S}
	m, n = size(X)
	WTW = HHT = Matrix{T}(undef, k ,k)
	gW = Vector{T}(undef, m)
	gH = Vector{T}(undef, n)

	H = transpose(HT)
	WT = transpose(W)
	XT = transpose(X)

	norm_pg0 = 0.0
	norm_pg_prev = Inf

	iter = 1
	converged = false
	while ! converged && iter < maxiter
		norm_pg = zero(T)

		# updateW
		mul!(HHT, H, HT)
		norm_pg += updateHALS!(gW, W, X, HT, HHT, alphaW)

		# updateH
		mul!(WTW, WT, W)
		norm_pg += updateHALS!(gH, HT, XT, W, WTW, alphaH)

		if iter == 1
			norm_pg0 = norm_pg
		else
			norm_pg = norm_pg / norm_pg0
		end

		converged = abs(norm_pg - norm_pg_prev) < tol
		norm_pg_prev = norm_pg

		iter += 1
	end

	converged
end

function nmf(::Type{T}, X::AbstractMatrix{S}, k::Int; tol=1e-4, maxiter=200, alphaW::L1L2{T}=zero(L1L2{T}), alphaH::L1L2{T}=zero(L1L2{T})) where {T,S}
	W, HT = initialize_nmf(Float64, X, k)
	converged = nmf!(W, HT, X, k; tol=tol, maxiter=maxiter, alphaW=alphaW, alphaH=alphaH)

	converged || @warn "failed to converge in $maxiter iterations"
	W, HT
end

function nmf(X::AbstractMatrix{S}, k::Int; tol=1e-4, maxiter=200, alphaW::L1L2{Float64}=zero(L1L2{Float64}), alphaH::L1L2{Float64}=zero(L1L2{Float64})) where S
  nmf(Float64, X, k; tol=tol, maxiter=maxiter, alphaW=alphaW, alphaH=alphaH)
end
