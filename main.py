
import tkinter as tk
from gui import NeuralNetworkGUI


def main():
    print("="*70)
    print("Penguin Classification - Perceptron & Adaline Neural Networks")
    print("="*70)
    print("\nStarting GUI application...")
    print("\nInstructions:")
    print("1. Select two features from the dropdown menus")
    print("2. Select two classes to classify")
    print("3. Set hyperparameters (learning rate, epochs, MSE threshold)")
    print("4. Choose algorithm (Perceptron or Adaline)")
    print("5. Click 'Train Model' to train")
    print("6. Click 'Test Model' to evaluate on test data")
    print("7. Click 'Visualize' to see decision boundary and learning curves")

    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
