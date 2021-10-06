using NMF, Test

@testset "initialization" begin
    m, n, k = 100, 10, 5
    X = abs.(randn(m, n))

    @testset "default" begin
        HT, W = initialize(X, k)
        @test size(HT) == (k, m)
        @test size(W) == (k, n)
        @test eltype(HT) == eltype(W) == Float64
    end

    @testset "dtype" begin
        dtype = Float32
        HT, W = initialize(X, k, dtype=dtype)
        @test size(HT) == (k, m)
        @test size(W) == (k, n)
        @test eltype(HT) == eltype(W) == dtype
    end
end
