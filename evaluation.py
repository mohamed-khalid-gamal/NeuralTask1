
import numpy as np


class Evaluator:

    @staticmethod
    def confusion_matrix(y_true, y_pred):


        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == -1) & (y_pred == -1))
        FP = np.sum((y_true == -1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == -1))

        return np.array([[TN, FP],
                        [FN, TP]])

    @staticmethod
    def accuracy(y_true, y_pred):

        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(confusion_matrix):
        TN, FP, FN, TP = confusion_matrix.ravel()
        if (TP + FP) == 0:
            return 0.0
        return TP / (TP + FP)

    @staticmethod
    def recall(confusion_matrix):
        TN, FP, FN, TP = confusion_matrix.ravel()
        if (TP + FN) == 0:
            return 0.0
        return TP / (TP + FN)

    @staticmethod
    def f1_score(confusion_matrix):
        prec = Evaluator.precision(confusion_matrix)
        rec = Evaluator.recall(confusion_matrix)

        if (prec + rec) == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    @staticmethod
    def print_metrics(y_true, y_pred, class1_name, class2_name):

        cm = Evaluator.confusion_matrix(y_true, y_pred)
        acc = Evaluator.accuracy(y_true, y_pred)

        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted {class1_name}  Predicted {class2_name}")
        print(f"Actual {class1_name:10s}      {cm[0,0]:6d}          {cm[0,1]:6d}")
        print(f"Actual {class2_name:10s}      {cm[1,0]:6d}          {cm[1,1]:6d}")

        print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {Evaluator.precision(cm):.4f}")
        print(f"Recall: {Evaluator.recall(cm):.4f}")
        print(f"F1 Score: {Evaluator.f1_score(cm):.4f}")
        print("="*50 + "\n")

        return cm, acc

