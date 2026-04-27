load("data/cifar-car-truck.RData")

png("debug_images.png", width=1200, height=400)

par(mfrow=c(2,4), mar=c(1,1,2,1))
for (i in 1:8) {
  img <- x.train[i,,,] / 255
  img <- aperm(img, c(2,1,3))
  plot(as.raster(img))
  title(paste("Label:", y.train[i]), cex.main=0.8)
}

dev.off()