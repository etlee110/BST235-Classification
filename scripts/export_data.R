load("data/cifar-car-truck.RData")

cat("Objects:", paste(ls(), collapse=", "), "\n")
cat("x.train dim:", paste(dim(x.train), collapse=" x "), "\n")
cat("x.test  dim:", paste(dim(x.test),  collapse=" x "), "\n")
cat("y.train length:", length(y.train), "\n")
cat("y.train unique:", paste(sort(unique(y.train)), collapse=", "), "\n")

# x.train is (N, x, y, RGB) in R's column-major layout.
# We write the raw floats; Python will reshape as Fortran order then transpose.
x_train_float <- array(as.numeric(x.train), dim=dim(x.train))
x_test_float  <- array(as.numeric(x.test),  dim=dim(x.test))

writeBin(as.vector(x_train_float), "data/x_train_raw.bin", size=4)
writeBin(as.vector(x_test_float),  "data/x_test_raw.bin",  size=4)
writeBin(as.integer(y.train),      "data/y_train_raw.bin", size=4)

cat("dims written — x_train:", paste(dim(x_train_float), collapse="x"),
    " x_test:", paste(dim(x_test_float), collapse="x"), "\n")
cat("x_train range:", min(x_train_float), max(x_train_float), "\n")
