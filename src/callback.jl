abstract type AbstractCallback end

# source: https://youtu.be/EpNeNCGmyZE?t=633
# default fucntion blabla(cb;kwargs...) and fucntion blabla!(cb;kwargs...)
#for sym in [:before_fit, :before_epoch, :before_all_batches, :before_batch, :after_batch, :before_validate, :after_all_batches, :after_epoch, :after_fit]
#    @eval function $sym(cb::AbstractCallback;kwargs...)
#    end
#end

for sym in [:before_fit, :before_epoch, :before_all_batches, :before_batch, :after_batch, :before_validate, :after_all_batches, :after_epoch, :after_fit]
    @eval function $(Symbol(string(sym,"!")))(cb::AbstractCallback;kwargs...)
    end
end

@with_kw mutable struct TrainEvalCallback <: AbstractCallback
    order::Int = -1
    n_epochs::Int = 0
    current_epoch::Float64 = 0.0
    n_batches::Int = 0 
    current_batch::Int = 0
    in_train::Bool = true
    use_cuda::Bool = false
    num_recorder::Int = 0
end

init_c_epoch!(cb::TrainEvalCallback) = cb.current_epoch = 0
init_c_batch!(cb::TrainEvalCallback) = cb.current_batch = 0
incr_c_batch!(cb::TrainEvalCallback) = cb.current_batch += 1
incr_c_epoch!(cb::TrainEvalCallback) = cb.current_epoch += 1. /cb.n_batches

function calculate_n_nbatches!(cb::TrainEvalCallback, d::DataLoader)
    cb.n_batches =  ceil(d.nobs / d.batchsize)
end

function before_fit!(cb::TrainEvalCallback; n_epochs::Int, model, loss_func, kwargs...)
    init_c_epoch!(cb)
    init_c_batch!(cb)
    cb.n_epochs = n_epochs
end

function after_batch!(cb::TrainEvalCallback; kwargs...)
    incr_c_batch!(cb)
    incr_c_epoch!(cb)
end

function before_epoch!(cb::TrainEvalCallback; epoch, kwargs...)
    cb.current_epoch = epoch-1# start at zero
    cb.in_train = true
end

function before_all_batches!(cb::TrainEvalCallback; d::DataLoader, kwargs...)
    calculate_n_nbatches!(cb,d)
end

function before_validate!(cb::TrainEvalCallback; kwargs...)
    cb.in_train = false
end

@with_kw mutable struct AvgStatsCallback{F<:Function} <: AbstractCallback
    order::Int = 0
    current_sum_stat::Float64 = 1000.0
    num_samples::Int = 0
    average_stat::Float64 = 1000.0
    f::F = logitcrossentropy
    name::String = "loss"
    recorder::Bool = true
    rec_ind::Int = 0
    rec_num::Int = 0
    recorder_initialzed::Bool = false
end

function AvgStatsCallback(order::Int,f::Function; kwargs...)
    AvgStatsCallback{typeof(f)}(order=order, f=f, name=String(Symbol(f)))
end

function before_fit!(cb::AvgStatsCallback; te_cb::TrainEvalCallback, kwargs...)
    if cb.recorder && !cb.recorder_initialzed
        te_cb.num_recorder += 1
        cb.rec_num = te_cb.num_recorder
        cb.recorder_initialzed = true
    end
end

function after_batch!(cb::AvgStatsCallback; rec, batch_loss, batch_size, batch_pred, batch_label, kwargs...)
    if cb.name == "loss"
         incr = batch_loss 
    else
        incr = cb.f(batch_pred,batch_label) 
    end
    cb.current_sum_stat += incr*batch_size
    cb.num_samples += batch_size
    if cb.recorder
        cb.rec_ind += 1
        rec[cb.rec_num,cb.rec_ind] = incr
    end
end

function after_all_batches!(cb::AvgStatsCallback; te_cb::TrainEvalCallback, epoch::Int, kwargs...)
    cb.average_stat = cb.current_sum_stat/cb.num_samples
    cb.current_sum_stat = 0
    cb.num_samples = 0
    if te_cb.in_train
        @assert round(Int, te_cb.current_epoch) == epoch "pb not passing over all the train set"
        println("Train - epoch: ", epoch, " ", cb.name, ": ", cb.average_stat)
    else
        @assert round(Int, te_cb.current_epoch) == epoch+1 "pb not passing over all the validation set"
        println("Val - epoch: ", epoch, " ", cb.name, ": ", cb.average_stat)
    end
end