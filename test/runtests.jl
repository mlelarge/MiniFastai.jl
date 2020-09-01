using MiniFastai
using Test

@testset "MiniFastai.jl" begin
    # Write your tests here.
    include("test_data.jl")
    include("test_callback.jl")
    include("test_cnn.jl")
    #include("test_learner.jl")
    include("test_training.jl")
end
