accuracy(pred,y) = mean(onecold(pred |>  cpu).==onecold(y |> cpu))