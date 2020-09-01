abstract type AbstractHook end

mutable struct Hook <: AbstractHook 
    layer_num::Int
    logg::Array{Real,1}
    count::Int
    function Hook(n,logg;count=0)
        new(n,logg,count)
    end
end

function logg!(x,h::Hook)
    h.count += 1
    h.logg[h.count] = mean(x)
    x
end

function hook_layer(x;h::Hook)
    return logg!(x,h)
end

#todo https://fluxml.ai/Zygote.jl/latest/adjoints/
@adjoint hook_layer(x; kwargs...) = hook_layer(x; kwargs...), b -> (b,)

function make_hooks(num_hooks, n)
    return Dict(i => Hook(i,zeros(n)) for i in num_hooks)
end

function make_new_layers(model, dic_hooks)
    t = ()
    for (i,l) in enumerate(model.layers)
        if haskey(dic_hooks,i)
            t = tuple(t..., l, x -> hook_layer(x,h=dic_hooks[i]))
        else
            t = tuple(t..., l)
        end
    end
    return t
end

function get_model_hooks(model,num_hooks;n=10)
    dic_hooks = make_hooks(num_hooks,n)
    n_layers = make_new_layers(model,dic_hooks)
    return (Chain(n_layers...), dic_hooks)
end