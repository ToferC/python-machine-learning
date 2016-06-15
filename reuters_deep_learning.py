from keras.datasets import reuters

(X_train, y_train), (X_test, y_test) = reuters.load_data(
	path="reuters.pkl",
	nb_words=None,
	skip_top=0,
	maxlen=None,
	test_split=0.1)