library(keras)
reuters <- dataset_reuters(num_words = 1000, test_split = 0.2)
x_train <- reuters$train$x
y_train <- reuters$train$y
x_test <- reuters$test$x
y_test <- reuters$test$y
num_classes <- count_params(y_train)

tokenizer <- text_tokenizer(num_words = 1000)
x_train <- sequences_to_matrix(tokenizer, x_train, mode = 'binary')
x_test <- sequences_to_matrix(tokenizer, x_test, mode = 'binary')