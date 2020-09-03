"""
    normalize(x,m,s)

Substract scalar `m` from array `x` and rescale by scalar `s`.
"""
function normalize(x,m,s)
    return (x.-m)./s
end

"""
    normalize_imgs(imgs, labels; is_train=true, m=0, s=1)

Take a onehot encoding of the labels.
If `is_train=true`, compute the mean and std of the images and normalize them. 
Return the normalized images and the mean and std computed.
Otherwise, normalize the imgs with `m` and `s`.
Should be used with `is_train=true` on the train set and with `is_train=false` 
otherwise with the parameters `m` and `s` computed on the training set.
"""
function normalize_imgs(imgs, labels; is_train=true, m=0, s=1)
    X = Array{Float32}(undef, size(imgs[1])..., 1, length(imgs))
    for i in 1:length(imgs)
        X[:, :, :, i] = Float32.(imgs[i])
    end
    Y = onehotbatch(labels, 0:9)
    if is_train
        m = mean(X)
        s = std(X)
        return (normalize(X,m,s), Y), m, s
    else
        return (normalize(X,m,s), Y)
    end
end

function get_processed_data_MNIST()
    # Load labels and images from Flux.Data.MNIST
    train_labels = MNIST.labels()
    train_imgs = MNIST.images()    
    train_set, m, s = normalize_imgs(train_imgs,train_labels,is_train=true)
    
    test_imgs = MNIST.images(:test)
    test_labels = MNIST.labels(:test)
    test_set = normalize_imgs(test_imgs,test_labels,is_train=false,m=m,s=s)
    return train_set, test_set
end

abstract type AbstractDatabunch end

"""
    Databunch(train_loader::Flux.Data.DataLoader,val_loader::Flux.Data.DataLoader)

Two dataloaders, one for the train set and one for the validation set.
"""
struct Databunch <: AbstractDatabunch
    train_loader::Flux.Data.DataLoader
    val_loader::Flux.Data.DataLoader
end