@testset "callback" begin
    a_cb = AvgStatsCallback()

    @test a_cb.order == 0
end