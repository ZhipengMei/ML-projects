import classifier
training_file = "../dataset/spam_train.txt"
testing_file = "../dataset/spam_test.txt"

classifier.email_spam_classifier(training_file, testing_file)
