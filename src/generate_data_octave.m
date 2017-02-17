%generate dataset from octave
[train_x, train_t, valid_x, valid_t, test_x, test_t, vocab] = load_data(372500);
csvwrite("train_x.csv",train_x')
csvwrite("train_t.csv",train_t')
csvwrite("valid_x.csv",valid_x')
csvwrite("valid_t.csv",valid_t')
csvwrite("test_x.csv",test_x')
csvwrite("test_t.csv",test_t')