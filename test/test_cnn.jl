@testset "cnn" begin
    nfs = [8,16,32,32]
    cnn_model = get_model(nfs)
    X = randn(Float32,(28,28,1,12))
    Y = cnn_model(X)

    @test size(Y) == (10,12)
end