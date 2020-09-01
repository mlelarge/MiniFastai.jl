module MiniFastai

using Flux
using Flux.Data: MNIST, DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using Flux.Optimise: update!, Descent
using Parameters: @with_kw
using Statistics: mean, std
using Zygote: pullback, @adjoint
using CUDA

include("data.jl")
include("callback.jl")
include("cnn.jl")
include("learner.jl")
include("hook.jl")
include("training.jl")

export my_f

export get_processed_data_MNIST
export AbstractDatabunch
export Databunch
export normalize
export normalize_imgs

export AbstractCallback
export TrainEvalCallback
export AvgStatsCallback

export get_model

export AbstractLearner
export Learner

export get_model_hooks

export fit!

end
