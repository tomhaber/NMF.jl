import Statistics: mean
import Random: randn

function objective(X::AbstractMatrix, W::AbstractMatrix, HT::AbstractMatrix)
	WH = W*HT'

	llik = 0.0
	@inbounds for i in eachindex(X)
		llik += X[i]*log((X[i]+1e-5)/(WH[i]+1e-15))-X[i]+WH[i];
	end

	llik
end

# g = (1 .- (X ./ (WH .+ 1e-10))) * HT[:,q]
# h = ((X ./ (WH .+ 1e-10).^2)) * HT[:,q].^2

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

function klnmf!(W::AbstractMatrix{T}, HT::AbstractMatrix{T}, X::AbstractMatrix{S}; tol=1e-4, maxiter=200) where {T,S}
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
