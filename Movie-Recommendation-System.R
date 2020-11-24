##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(digest)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


## Using User and Movie Effect only on the validation set.

library(recosystem)
set.seed(1)


lambda <- 4.75

mu <- mean(edx$rating)

Beta_movie <- edx %>%
  group_by(movieId) %>%
  summarize(Beta_movie = sum(rating - mu)/(n()+lambda)) %>%
  ungroup()

Beta_user <- edx %>%
  left_join(Beta_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarize(Beta_user = sum(rating - Beta_movie - mu)/(n()+lambda)) %>%
  ungroup()


predicted_ratings <- validation %>% 
  left_join(Beta_movie, by = "movieId") %>%
  left_join(Beta_user, by = "userId") %>%
  mutate(pred = mu + Beta_movie + Beta_user) %>%
  pull(pred)

residual <- edx %>% 
  left_join(Beta_movie, by = "movieId") %>%
  left_join(Beta_user, by = "userId") %>%
  mutate(residual = rating - mu - Beta_movie - Beta_user) %>%
  select(userId, movieId, residual)


edx_set_residual_matrix <- as.matrix(residual)

validation_set_matrix <- validation %>% 
  select(userId, movieId, rating) %>% as.matrix(.)

write.table(edx_set_residual_matrix , file = "edx.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(validation_set_matrix, file = "validation.txt" , sep = " " , row.names = FALSE, col.names = FALSE)


# use data_file() to specify a data set from a file in the hard disk.

set.seed(1) 
edx_set <- data_file("edx.txt")
validation_set <- data_file("validation.txt")

# build a recommender object

r <-Reco()

# tuning training set (edx_set)

opts <- r$tune(edx_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                    costp_l1 = 0, costq_l1 = 0,
                                    nthread = 1, niter = 10))
opts


# training the recommender model

r$train(edx_set, opts = c(opts$min, nthread = 1, niter = 50))

# Making prediction on validation set and calculating RMSE:

predictions_file <- tempfile()
r$predict(validation_set, out_file(predictions_file))  
mf_forecasted_residuals <- scan(predictions_file)
mf_forecasted_ratings <- predicted_ratings + mf_forecasted_residuals
rmse_mf <- RMSE(mf_forecasted_ratings, validation %>% pull(rating)) 

final_rmse_results <- data.frame(Method = "Matrix Factorization", RMSE = rmse_mf)
final_rmse_results




