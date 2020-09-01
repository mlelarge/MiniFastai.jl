function fit!(l::Learner, n_epochs)
    mod = model(l)
    los_f = loss_func(l)
    opt = optimizer(l)
    dat = data(l)
    r = trainevalcb(l)
    cb_before_fit!(l, n_epochs=n_epochs, model=mod, loss_func=los_f)
    if r.use_cuda
        mod = mod |> gpu
        los_f = los_f |> gpu
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
            current_loss, back_loss = pullback(y -> loss(y,yb),y_pred)
            #current_loss, back = pullback(() -> los_f(mod(xb),yb), ps)
            gs = back_net(back_loss(1)[1])#back(one(current_loss))
            update!(SGD,ps,gs)
        else
            y_pred = mod(xb)
            current_loss = los_f(y_pred,yb)
        end
        cb_after_batch!(l, batch_loss=current_loss, batch_pred = y_pred, batch_label = yb, batch_size=size(xb,1))
    end
    
    function all_batches(dat)
        if r.in_train
            data_loader = dat.train_loader
        else
            data_loader = dat.val_loader
        end
        cb_before_all_batches!(l, d=data_loader)
        for (i, (xb,yb)) in enumerate(data_loader)
            one_batch(xb,yb)
        end
        cb_after_all_batches!(l)
    end
    
    for epoch in 1:n_epochs
        cb_before_epoch!(l, epoch=epoch-1)
        all_batches(dat)
        cb_before_validate!(l)
        if !r.in_train
            all_batches(dat)
        end
        cb_after_epoch!(l, epoch=epoch)
    end
    cb_after_fit!(l)
end