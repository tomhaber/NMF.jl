using NMF, Test
import SparseArrays: sprand, sparse, dropzeros!
import Distributions: Poisson, Normal, rand

@testset "measures" begin
    @testset "l2" begin
        X = sprand(100, 10, .1, (i) -> rand(Normal(4), i))
        Z = Matrix(X)

        HT, W = initialize(X, 5)
        meas = L2NMF()

        @test objective(meas, X, HT, W) ≈ objective(meas, Z, HT, W)
    end

    @testset "kl" begin
        X = dropzeros!(sprand(100, 10, .1, (i) -> rand(Poisson(4), i)))
        Z = Matrix(X)

        HT, W = initialize(X, 5)
        meas = KLNMF()

        @test objective(meas, X, HT, W) ≈ objective(meas, Z, HT, W)
    end

    @testset "regularized" begin
        X = dropzeros!(sprand(100, 10, .1, (i) -> rand(Poisson(4), i)))
        Z = Matrix(X)

        HT, W = initialize(X, 5)
        meas = Regularized(KLNMF(), alphaW=L1L2(1.0, 0.0), alphaH=L1L2(0.5, 0.1))

        obj = objective(KLNMF(), X, HT, W) + norm(W,1) + norm(HT, 2) * 0.1 + norm(HT,1) * 0.5
        @test objective(meas, X, HT, W) ≈ obj
        @test objective(meas, Z, HT, W) ≈ obj
    end
end
