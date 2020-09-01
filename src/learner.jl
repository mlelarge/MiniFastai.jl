abstract type AbstractLearner end

mutable struct Learner <: AbstractLearner
    model
    optimizer
    loss_func
    data::AbstractDatabunch
    callbacks
    function Learner(model,optimizer,loss_func,data;cbs=())
        cb_te = TrainEvalCallback()
        all_cbs = (cb_te,cbs...)
        ordered_cbs = Tuple(z[2] for z in sort([(cb.order, cb) for cb in all_cbs], by = x -> x[1]))
        new(model, optimizer, loss_func, data, ordered_cbs)
    end
end

model(l::Learner) = l.model
#model!(l::Learner, model) = l.model = model

optimizer(l::Learner) = l.optimizer
#optimizer!(l::Learner, optimizer) = l.optimizer = optimizer

loss_func(l::Learner) = l.loss_func
#loss_func!(l::Learner, loss_func) = l.loss_func = loss_func

data(l::Learner) = l.data
#data!(l::Learner, data) = l.data = data

callbacks(l::Learner) = l.callbacks
#callbacks!(l::Learner, callbacks) = l.callbacks = callbacks

trainevalcb(l::Learner) = l.callbacks[1]

# default function cb_blabla!(l::AbstractLearner;kwargs...)
for sym in [:before_fit, :before_epoch, :before_all_batches, :before_batch, :after_batch, :before_validate, :after_validate, :after_all_batches, :after_epoch, :after_fit]
    @eval function $(Symbol(string("cb_",sym,"!")))(l::AbstractLearner;kwargs...)
        te_cb = trainevalcb(l)
        for cb in l.callbacks
           $(Symbol(string(sym,"!")))(cb;te_cb=te_cb, kwargs...) 
        end
    end
end