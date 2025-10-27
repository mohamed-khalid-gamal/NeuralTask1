
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    @staticmethod
    def plot_decision_boundary(X_train, y_train, X_test, y_test, model,
                               feature1_name, feature2_name,
                               class1_name, class2_name,
                               algorithm_name, title=None):

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        Visualizer._plot_data_with_boundary(
            X_train, y_train, model, feature1_name, feature2_name,
            class1_name, class2_name, "Training Data"
        )

        plt.subplot(1, 2, 2)
        Visualizer._plot_data_with_boundary(
            X_test, y_test, model, feature1_name, feature2_name,
            class1_name, class2_name, "Test Data"
        )

        if title is None:
            title = f"{algorithm_name}: {class1_name} vs {class2_name}\n{feature1_name} vs {feature2_name}"
        plt.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def _plot_data_with_boundary(X, y, model, feature1_name, feature2_name,
                                  class1_name, class2_name, subtitle):
        class1_mask = (y == -1)
        class2_mask = (y == 1)

        plt.scatter(X[class1_mask, 0], X[class1_mask, 1],
                   c='blue', marker='o', s=100, alpha=0.7,
                   edgecolors='black', label=class1_name)
        plt.scatter(X[class2_mask, 0], X[class2_mask, 1],
                   c='red', marker='s', s=100, alpha=0.7,
                   edgecolors='black', label=class2_name)

        w0, w1, bias = model.get_decision_boundary()

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_vals = np.array([x1_min, x1_max])

        if abs(w1) > 1e-10:
            x2_vals = -(w0 * x1_vals + bias) / w1
            plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='Decision Boundary')
        else:
            x1_boundary = -bias / w0 if abs(w0) > 1e-10 else 0
            plt.axvline(x=x1_boundary, color='k', linewidth=2, label='Decision Boundary')

        plt.xlabel(feature1_name, fontsize=11)
        plt.ylabel(feature2_name, fontsize=11)
        plt.title(subtitle, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def plot_learning_curve(model, algorithm_name):

        plt.figure(figsize=(8, 5))

        if hasattr(model, 'errors_per_epoch'):
            epochs = range(1, len(model.errors_per_epoch) + 1)
            plt.plot(epochs, model.errors_per_epoch, 'b-o', linewidth=2)
            plt.ylabel('Number of Misclassifications', fontsize=11)
            plt.title(f'{algorithm_name} Learning Curve', fontsize=13, fontweight='bold')
        elif hasattr(model, 'mse_per_epoch'):
            epochs = range(1, len(model.mse_per_epoch) + 1)
            plt.plot(epochs, model.mse_per_epoch, 'r-o', linewidth=2)
            plt.ylabel('Mean Squared Error (MSE)', fontsize=11)
            plt.title(f'{algorithm_name} Learning Curve', fontsize=13, fontweight='bold')

        plt.xlabel('Epoch', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_confusion_matrix(cm, class1_name, class2_name, algorithm_name):

        plt.figure(figsize=(6, 5))

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {algorithm_name}', fontsize=13, fontweight='bold')
        plt.colorbar()

        classes = [class1_name, class2_name]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=11)
        plt.yticks(tick_marks, classes, fontsize=11)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16, fontweight='bold')

        plt.ylabel('True Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()
        return plt.gcf()
