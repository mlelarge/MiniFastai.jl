function param_conv2d(ni,no;k=3,stride=2)
    return Dict("filter" => (k,k), "in" => ni, "out" => no, "padding" => k÷2, "stride" => stride)
end

function make_conv(d::Dict)
    return Conv(d["filter"], d["in"] => d["out"], relu, stride = d["stride"], pad = d["padding"])
end

function make_list_i_o(nfs;start=1)
    return [i==0 ? (start,nfs[1]) : (nfs[i],nfs[i+1]) for i in 0:length(nfs)-1]
end

function make_list_layers(nfs)
    conv_layers = Tuple(i==1 ? make_conv(param_conv2d(i,o,k=5)) : make_conv(param_conv2d(i,o)) for (i,o) in make_list_i_o(nfs))
    return tuple(conv_layers..., AdaptiveMeanPool((1,1)), flatten, Dense(last(nfs), 10))
end

function get_model(nfs)
    layers = make_list_layers(nfs)
    return Chain(layers...)
end 
