function fit!(l::Learner, n_epochs)
    mod = model(l)
    los_f = loss_func(l)
    opt = optimizer(l)
    dat = data(l)
    r = trainevalcb(l)
    cb_before_fit!(l, te_cb=r, n_epochs=n_epochs, model=mod, loss_func=los_f)
    if r.use_cuda
        println("using GPU")
        mod = mod |> gpu
        los_f = los_f |> gpu
    end
    if r.num_recorder > 0
        rec = Array{Float64}(undef,r.num_recorder,n_epochs*(length(dat.train_loader)+length(dat.val_loader)))
    end

    function one_batch(xb,yb)
        if r.use_cuda
            xb, yb = xb |> gpu, yb |> gpu
        end
        cb_before_batch!(l, xb=xb, yb =yb)
        #y_pred = mod(xb)
        #current_loss = los_f(y_pred,yb)
        if r.in_train
            ps = params(mod)
            #source https://fluxml.ai/Flux.jl/stable/training/training/
            # https://fluxml.ai/Zygote.jl/latest/adjoints/#Pullbacks-1
            y_pred, back_net = pullback(() -> mod(xb), ps)
            current_loss, back_loss = pullback(y -> los_f(y,yb),y_pred)
            #current_loss, back = pullback(() -> los_f(mod(xb),yb), ps)
            gs = back_net(back_loss(1)[1])#back(one(current_loss))
            update!(opt,ps,gs)
        else
            y_pred = mod(xb)
            current_loss = los_f(y_pred,yb)
        end
        cb_after_batch!(l, rec=rec, batch_loss=current_loss, batch_pred = y_pred, batch_label = yb, batch_size=size(xb,1))
    end
    
    function all_batches(dat,epoch)
        if r.in_train
            data_loader = dat.train_loader
        else
            data_loader = dat.val_loader
        end
        cb_before_all_batches!(l, d=data_loader)
        for (i, (xb,yb)) in enumerate(data_loader)
            one_batch(xb,yb)
        end
        cb_after_all_batches!(l,epoch=epoch)
    end
    
    for epoch in 1:n_epochs
        cb_before_epoch!(l, epoch=epoch)
        all_batches(dat,epoch)
        cb_before_validate!(l)
        if !r.in_train
            all_batches(dat,epoch)
        end
        cb_after_epoch!(l, epoch=epoch)
    end
    cb_after_fit!(l)
    return rec
end