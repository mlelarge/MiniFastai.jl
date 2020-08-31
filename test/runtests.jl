using MiniFastai
using Test

@testset "MiniFastai.jl" begin
    # Write your tests here.
    @test my_f(2,1) == 7
    @test my_f(2,3) == 13
end
