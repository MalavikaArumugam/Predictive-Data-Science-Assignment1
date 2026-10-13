data <- read.csv("datasets/breast_cancer_sample.csv", stringsAsFactors=FALSE)
data$diagnosis <- factor(data$diagnosis, levels=c("benign","malignant")); f <- c("radius","texture","smoothness")
centroids <- aggregate(data[,f], by=list(diagnosis=data$diagnosis), FUN=mean)
predict_one <- function(x){ d <- sapply(seq_len(nrow(centroids)), function(i) sum((as.numeric(x[1,f])-as.numeric(centroids[i,f]))^2)); as.character(centroids$diagnosis[which.min(d)]) }
p <- vapply(seq_len(nrow(data)), function(i) predict_one(data[i,,drop=FALSE]), character(1))
cat("BREAST CANCER CLASSIFICATION — TEACHING DEMONSTRATION\n"); cat("Number of records:",nrow(data),"\n"); cat(sprintf("Resubstitution accuracy: %.1f%%\n",mean(p==as.character(data$diagnosis))*100)); cat("Example record prediction:",predict_one(data.frame(radius=20,texture=26,smoothness=.17)),"\n"); cat("WARNING: Tiny illustrative data; not for diagnosis.\n")
