import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	fig, (axe1, axe2) = plt.subplots(1, 2)
	axe1.plot(train_losses, 'r-', label='training loss')
	axe1.plot(valid_losses, 'b-', label='validation loss')
	axe1.set_title("Loss Curve")
	axe1.legend(loc="upper right")
	axe1.set_xlabel("epoch")
	axe1.set_ylabel("Loss")

	axe2.plot(train_accuracies, 'r-', label='training accuracy')
	axe2.plot(valid_accuracies, 'g-', label='validation accuracy')
	axe2.legend(loc="upper left")
	axe2.set_title("Accuracy Curve")
	axe2.set_xlabel("epoch")
	axe2.set_ylabel("Accuracy")
	# plt.show()
	fig.savefig('myrnn_lc.png')


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	y_true, y_pred = zip(*results)
	cm = confusion_matrix(y_true, y_pred, normalize='true')
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
	disp.plot(cmap='Blues')
	plt.title("Normalized Confusion Matrix")
	# plt.show()
	plt.savefig('myrnn_cm.png')
