imdb <- dataset_imdb(num_words = 5000)
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = 400)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = 400)
y_train <- imdb$train$y
y_test <- imdb$test$y

model2 <- keras_model_sequential()
model2 %>%
  layer_embedding(5000, 50, input_length = 400) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(250, 3, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(250) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history2 <- model2 %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 4,
  verbose = 1,
  validation_split = 0.1
)