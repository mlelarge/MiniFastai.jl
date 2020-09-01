using Flux.Data
using Flux.Optimise
using Flux: logitcrossentropy
@testset "training" begin
    all_train_set, _ = get_processed_data_MNIST()
    train_set = (all_train_set[1][:,:,:,1:1000], all_train_set[2][:,1:1000])
    val_set = (all_train_set[1][:,:,:,1000:2000], all_train_set[2][:,1000:2000])
    bs=512
    train_loader = DataLoader(train_set, batchsize=bs, shuffle=true)
    val_loader = DataLoader(val_set, batchsize=bs, shuffle=false)
    databunch = Databunch(train_loader,val_loader)
    nfs = [8,16,32,32]
    cnn_model = get_model(nfs)
    num_hooks = [1,2,3,4]
    n_epochs = 2
    (cnn_model_hook, dic_hooks) = get_model_hooks(cnn_model,num_hooks,n=n_epochs*(length(train_loader)+length(val_loader)))
    learning_rate = 1e-1
    SGD = Descent(learning_rate)
    loss(pred,y) = logitcrossentropy(pred,y)
    learner_cnn = Learner(cnn_model_hook,SGD,loss,databunch,cbs=(AvgStatsCallback(),AvgStatsCallback(order=1,name=:accuracy),))

    fit!(learner_cnn,n_epochs)
end