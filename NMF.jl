module NMF
	import Statistics: mean
	import Random: rand!, rand, randn!, seed!
	import LinearAlgebra: axpby!, axpy!, mul!, dot, norm

	L1L2{T <: Real} = Tuple{T, T}
	Base.zero(::Type{L1L2{T}}) where {T <: Real} = (zero(T), zero(T))

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

	include("nmf.jl")
	include("klnmf.jl")

	export nmf, nmf!
	export klnmf, klnmf!
	export L1L2
end
