var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MiniFastai","category":"page"},{"location":"#MiniFastai","page":"Home","title":"MiniFastai","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MiniFastai]","category":"page"},{"location":"#MiniFastai.Databunch","page":"Home","title":"MiniFastai.Databunch","text":"Databunch(train_loader::Flux.Data.DataLoader,val_loader::Flux.Data.DataLoader)\n\nTwo dataloaders, one for the train set and one for the validation set.\n\n\n\n\n\n","category":"type"},{"location":"#MiniFastai.normalize-Tuple{Any, Any, Any}","page":"Home","title":"MiniFastai.normalize","text":"normalize(x,m,s)\n\nSubstract scalar m from array x and rescale by scalar s.\n\n\n\n\n\n","category":"method"},{"location":"#MiniFastai.normalize_imgs-Tuple{Any, Any}","page":"Home","title":"MiniFastai.normalize_imgs","text":"normalize_imgs(imgs, labels; is_train=true, m=0, s=1)\n\nTake a onehot encoding of the labels. If is_train=true, compute the mean and std of the images and normalize them.  Return the normalized images and the mean and std computed. Otherwise, normalize the imgs with m and s. Should be used with is_train=true on the train set and with is_train=false  otherwise with the parameters m and s computed on the training set.\n\n\n\n\n\n","category":"method"}]
}
