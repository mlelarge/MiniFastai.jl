@testset "data" begin
    @test normalize([1,2],0.5,0.5) == [1.0,3.0]
end