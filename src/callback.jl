abstract type AbstractCallback end

# source: https://youtu.be/EpNeNCGmyZE?t=633
# default fucntion blabla(cb;kwargs...) and fucntion blabla!(cb;kwargs...)
for sym in [:before_fit, :before_epoch, :before_all_batches, :before_batch, :after_batch, :before_validate, :after_all_batches, :after_epoch, :after_fit]
    @eval function $sym(cb::AbstractCallback;kwargs...)
    end
end

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
    cb.current_epoch = epoch
    cb.in_train = true
end

function before_all_batches!(cb::TrainEvalCallback; d::DataLoader, kwargs...)
    calculate_n_nbatches!(cb,d)
end

function before_validate!(cb::TrainEvalCallback; kwargs...)
    cb.in_train = false
end

@with_kw mutable struct AvgStatsCallback <: AbstractCallback
    order::Int = 0
    current_sum_stat::Float64 = 1000.0
    num_samples::Int = 0
    average_stat::Float64 = 1000.0
    name = :loss
end

macro encaps(name_func, p, y)
    quote
        $(esc(name_func))($(esc(p)),$(esc(y)))
    end
end

function after_batch!(cb::AvgStatsCallback; batch_loss, batch_size, batch_pred, batch_label, kwargs...)
    if cb.name == :loss
        cb.current_sum_stat += batch_loss *batch_size
    else
        cb.current_sum_stat += @encaps(eval(cb.name),batch_pred,batch_label) *batch_size
    end
    cb.num_samples += batch_size
end

function after_all_batches!(cb::AvgStatsCallback; te_cb::TrainEvalCallback, kwargs...)
    cb.average_stat = cb.current_sum_stat/cb.num_samples
    cb.current_sum_stat = 0
    cb.num_samples = 0
    if te_cb.in_train
        println("Train - epoch: ", round(Int, te_cb.current_epoch), " ", cb.name, ": ", cb.average_stat)
    else
        println("Val - epoch: ", round(Int, te_cb.current_epoch-1), " ", cb.name, ": ", cb.average_stat)
    end
end