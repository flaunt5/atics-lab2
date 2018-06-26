library(keras)
reuters <- dataset_reuters(num_words = 1000, test_split = 0.2)
x_train <- reuters$train$x
y_train <- reuters$train$y
x_test <- reuters$test$x
y_test <- reuters$test$y

num_classes <- max(y_train) + 1

y_train <- to_categorical(reuters$train$y, num_classes)
y_test <- to_categorical(reuters$test$y, num_classes)

tokenizer <- text_tokenizer(num_words = 1000)
x_train <- sequences_to_matrix(tokenizer, x_train, mode = 'binary')
x_test <- sequences_to_matrix(tokenizer, x_test, mode = 'binary')

model1 <- keras_model_sequential() %>%
  layer_dense(units = 512, input_shape = c(1000), activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

model1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history1 <- model1 %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 5,
  verbose = 1,
  validation_split = 0.1
)