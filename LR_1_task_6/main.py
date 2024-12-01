from utils import load_data, split_data, evaluate_model
from classifier import train_svm, train_nb

data = load_data('data_multivar_nb.txt')

x_train, x_test, y_train, y_test = split_data(data)

svm_classifier = train_svm(x_train, y_train)
nb_classifier = train_nb(x_train, y_train)

svm_pred = svm_classifier.predict(x_test)
nb_pred = nb_classifier.predict(x_test)

svm_accuracy, svm_conf_matrix, svm_class_report = evaluate_model(y_test, svm_pred)
nb_accuracy, nb_conf_matrix, nb_class_report = evaluate_model(y_test, nb_pred)

print("SVM Accuracy:", svm_accuracy)
print("Naive Bayes Accuracy:", nb_accuracy)

print("\nSVM Confusion Matrix:\n", svm_conf_matrix)
print("\nNaive Bayes Confusion Matrix:\n", nb_conf_matrix)

print("\nSVM Classification Report:\n", svm_class_report)
print("\nNaive Bayes Classification Report:\n", nb_class_report)
