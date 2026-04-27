  # debug_cifar.R

  # install.packages(c("R.matlab", "gridExtra")) if needed
  #library(R.matlab)
  library(grid)
  library(gridExtra)

  # load .RData or .rda
  load("data/cifar-car-truck.RData")  # change filename

  # inspect objects
  print(ls())

  # assume data is in X and labels in y (adjust if different)
  # X expected shape: N x 3072 (flattened CIFAR)
  # y expected: N

  # check structure
  str(x.train)
  str(y.train)

  # function to reconstruct one image
  reconstruct_image <- function(x) {
    # x: length 3072
    r <- matrix(x[1:1024], nrow=32, byrow=TRUE)
    g <- matrix(x[1025:2048], nrow=32, byrow=TRUE)
    b <- matrix(x[2049:3072], nrow=32, byrow=TRUE)
    img <- array(0, dim=c(32,32,3))
    img[,,1] <- r
    img[,,2] <- g
    img[,,3] <- b
    img / max(img)  # normalize for display
  }

  # plot a few images
  plots <- list()
  for (i in 1:8) {
    img <- reconstruct_image(x.train[i, ])
    plots[[i]] <- rasterGrob(img)
  }

  grid.arrange(grobs=plots, ncol=8)