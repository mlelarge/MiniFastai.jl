@testset "learner" begin
    nfs = [8,16,32,32]
    model = get_model(nfs)
    opti = Descent(1e-2)
    loss(pred,y) = logitcrossentropy(pred,y)
    learn = Learner(model,opti,loss,data)

    @test learn.callbaks[1].order == 0
end