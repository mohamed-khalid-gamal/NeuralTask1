import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

from data_preprocessing import DataPreprocessor
from perceptron import Perceptron
from adaline import Adaline
from evaluation import Evaluator
from visualization import Visualizer


class NeuralNetworkGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Penguin Classification - Perceptron & Adaline")
        self.root.geometry("900x700")

        self.preprocessor = DataPreprocessor()
        self.preprocessor.preprocess()

        self.features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        self.classes = ['Adelie', 'Chinstrap', 'Gentoo']

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.setup_ui()

    def setup_ui(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky='nsew')

        title_label = ttk.Label(main_frame, text="Penguin Species Classification",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        feature_frame = ttk.LabelFrame(main_frame, text="Feature Selection", padding="10")
        feature_frame.grid(row=1, column=0, columnspan=2, sticky='we', pady=5)

        ttk.Label(feature_frame, text="Feature 1:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.feature1_var = tk.StringVar(value=self.features[0])
        feature1_combo = ttk.Combobox(feature_frame, textvariable=self.feature1_var,
                                      values=self.features, state='readonly', width=20)
        feature1_combo.grid(row=0, column=1, padx=5, pady=3)

        ttk.Label(feature_frame, text="Feature 2:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.feature2_var = tk.StringVar(value=self.features[1])
        feature2_combo = ttk.Combobox(feature_frame, textvariable=self.feature2_var,
                                      values=self.features, state='readonly', width=20)
        feature2_combo.grid(row=1, column=1, padx=5, pady=3)

        class_frame = ttk.LabelFrame(main_frame, text="Class Selection", padding="10")
        class_frame.grid(row=2, column=0, columnspan=2, sticky='we', pady=5)

        ttk.Label(class_frame, text="Class 1:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.class1_var = tk.StringVar(value=self.classes[0])
        class1_combo = ttk.Combobox(class_frame, textvariable=self.class1_var,
                                    values=self.classes, state='readonly', width=20)
        class1_combo.grid(row=0, column=1, padx=5, pady=3)

        ttk.Label(class_frame, text="Class 2:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.class2_var = tk.StringVar(value=self.classes[1])
        class2_combo = ttk.Combobox(class_frame, textvariable=self.class2_var,
                                    values=self.classes, state='readonly', width=20)
        class2_combo.grid(row=1, column=1, padx=5, pady=3)

        param_frame = ttk.LabelFrame(main_frame, text="Hyperparameters", padding="10")
        param_frame.grid(row=3, column=0, columnspan=2, sticky='we', pady=5)

        ttk.Label(param_frame, text="Learning Rate (η):").grid(row=0, column=0, sticky=tk.W, padx=5)
        # Default to Perceptron-friendly learning rate; will adjust on algorithm change
        self.eta_var = tk.StringVar(value="0.01")
        eta_entry = ttk.Entry(param_frame, textvariable=self.eta_var, width=22)
        eta_entry.grid(row=0, column=1, padx=5, pady=3)

        ttk.Label(param_frame, text="Number of Epochs (m):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.epochs_var = tk.StringVar(value="100")
        epochs_entry = ttk.Entry(param_frame, textvariable=self.epochs_var, width=22)
        epochs_entry.grid(row=1, column=1, padx=5, pady=3)

        ttk.Label(param_frame, text="MSE Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.mse_var = tk.StringVar(value="0.5")
        mse_entry = ttk.Entry(param_frame, textvariable=self.mse_var, width=22)
        mse_entry.grid(row=2, column=1, padx=5, pady=3)

        self.bias_var = tk.BooleanVar(value=True)
        bias_check = ttk.Checkbutton(param_frame, text="Add Bias", variable=self.bias_var)
        bias_check.grid(row=3, column=0, columnspan=2, pady=5)

        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm", padding="10")
        algo_frame.grid(row=4, column=0, columnspan=2, sticky='we', pady=5)

        self.algorithm_var = tk.StringVar(value="Perceptron")
        perceptron_radio = ttk.Radiobutton(algo_frame, text="Perceptron",
                                          variable=self.algorithm_var, value="Perceptron",
                                          command=self.on_algorithm_change)
        perceptron_radio.grid(row=0, column=0, padx=20, pady=5)

        adaline_radio = ttk.Radiobutton(algo_frame, text="Adaline",
                                       variable=self.algorithm_var, value="Adaline",
                                       command=self.on_algorithm_change)
        adaline_radio.grid(row=0, column=1, padx=20, pady=5)

        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)

        train_btn = ttk.Button(button_frame, text="Train Model", command=self.train_model, width=20)
        train_btn.grid(row=0, column=0, padx=10)

        test_btn = ttk.Button(button_frame, text="Test Model", command=self.test_model, width=20)
        test_btn.grid(row=0, column=1, padx=10)

        visualize_btn = ttk.Button(button_frame, text="Visualize", command=self.visualize, width=20)
        visualize_btn.grid(row=0, column=2, padx=10)



        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=7, column=0, columnspan=2, sticky='we', pady=5)

    def on_algorithm_change(self):
        algo = self.algorithm_var.get()
        if algo == "Perceptron":
            self.eta_var.set("0.01")
        else:
            self.eta_var.set("0.001")
        self.mse_var.set("0.5")

    def validate_inputs(self):
        if self.feature1_var.get() == self.feature2_var.get():
            messagebox.showerror("Error", "Please select different features!")
            return False

        if self.class1_var.get() == self.class2_var.get():
            messagebox.showerror("Error", "Please select different classes!")
            return False

        try:
            eta = float(self.eta_var.get())
            epochs = int(self.epochs_var.get())
            mse = float(self.mse_var.get())

            if eta <= 0 or epochs <= 0 or mse <= 0:
                raise ValueError("Values must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid hyperparameter values: {str(e)}")
            return False

        return True

    def train_model(self):
        if not self.validate_inputs():
            return

        try:
            feature1 = self.feature1_var.get()
            feature2 = self.feature2_var.get()
            class1 = self.class1_var.get()
            class2 = self.class2_var.get()
            algorithm = self.algorithm_var.get()

            self.status_var.set(f"Loading data: {class1} vs {class2}, {feature1} vs {feature2}...")
            self.root.update()

            normalize = (algorithm == "Adaline")
            X, y = self.preprocessor.get_class_data(class1, class2, feature1, feature2, normalize=normalize)
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.preprocessor.split_data(X, y, train_size=30, random_state=42)

            eta = float(self.eta_var.get())
            epochs = int(self.epochs_var.get())
            mse_threshold = float(self.mse_var.get())
            use_bias = self.bias_var.get()

            self.status_var.set(f"Training {algorithm}...")
            self.root.update()

            if algorithm == "Perceptron":
                self.model = Perceptron(learning_rate=eta, n_epochs=epochs,
                                       use_bias=use_bias, random_state=42)
            else:
                self.model = Adaline(learning_rate=eta, n_epochs=epochs,
                                    mse_threshold=mse_threshold, use_bias=use_bias,
                                    random_state=42)

            self.model.fit(self.X_train, self.y_train)

            y_train_pred = self.model.predict(self.X_train)
            train_acc = Evaluator.accuracy(self.y_train, y_train_pred)

            self.status_var.set(f"{algorithm} trained! Training Accuracy: {train_acc*100:.2f}%")
            messagebox.showinfo("Success",
                              f"{algorithm} training completed!\n"
                              f"Training Accuracy: {train_acc*100:.2f}%\n"
                              f"Epochs completed: {len(self.model.errors_per_epoch if algorithm=='Perceptron' else self.model.mse_per_epoch)}")

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Training failed")

    def test_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return

        try:
            y_test_pred = self.model.predict(self.X_test)

            class1 = self.class1_var.get()
            class2 = self.class2_var.get()

            cm, acc = Evaluator.print_metrics(self.y_test, y_test_pred, class1, class2)

            self.status_var.set(f"Test Accuracy: {acc*100:.2f}%")

            fig = Visualizer.plot_confusion_matrix(cm, class1, class2, self.algorithm_var.get())
            plt.show()

            messagebox.showinfo("Test Results",
                              f"Test Accuracy: {acc*100:.2f}%\n"
                              f"See console for detailed metrics.")

        except Exception as e:
            messagebox.showerror("Error", f"Testing failed: {str(e)}")

    def visualize(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return

        try:
            fig1 = Visualizer.plot_decision_boundary(
                self.X_train, self.y_train, self.X_test, self.y_test,
                self.model, self.feature1_var.get(), self.feature2_var.get(),
                self.class1_var.get(), self.class2_var.get(),
                self.algorithm_var.get()
            )

            fig2 = Visualizer.plot_learning_curve(self.model, self.algorithm_var.get())

            plt.show()

            self.status_var.set("Visualization displayed")

        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")



def main():
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
