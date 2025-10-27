
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from perceptron import Perceptron
from adaline import Adaline
from evaluation import Evaluator
from visualization import Visualizer
import os


class ReportGenerator:

    def __init__(self, output_dir='report_outputs'):
        self.preprocessor = DataPreprocessor()
        self.preprocessor.preprocess()
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.results = []

    def test_combination(self, class1, class2, feature1, feature2, algorithm='Perceptron',
                        eta=0.01, epochs=100, mse_threshold=0.01, use_bias=True):

        print(f"\n{'='*70}")
        print(f"Testing: {algorithm}")
        print(f"Classes: {class1} vs {class2}")
        print(f"Features: {feature1} vs {feature2}")
        print(f"{'='*70}")

        normalize = (algorithm == 'Adaline')
        X, y = self.preprocessor.get_class_data(class1, class2, feature1, feature2, normalize=normalize)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, train_size=30, random_state=42)

        if algorithm == 'Perceptron':
            model = Perceptron(learning_rate=eta, n_epochs=epochs,
                             use_bias=use_bias, random_state=42)
        else:
            model = Adaline(learning_rate=eta, n_epochs=epochs,
                          mse_threshold=mse_threshold, use_bias=use_bias,
                          random_state=42)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = Evaluator.accuracy(y_train, y_train_pred)
        test_acc = Evaluator.accuracy(y_test, y_test_pred)

        cm_train = Evaluator.confusion_matrix(y_train, y_train_pred)
        cm_test = Evaluator.confusion_matrix(y_test, y_test_pred)

        print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        combo_name = f"{algorithm}_{class1}_{class2}_{feature1}_{feature2}"

        fig1 = Visualizer.plot_decision_boundary(
            X_train, y_train, X_test, y_test, model,
            feature1, feature2, class1, class2, algorithm
        )
        fig1.savefig(f"{self.output_dir}/{combo_name}_boundary.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)

        fig2 = Visualizer.plot_learning_curve(model, algorithm)
        fig2.savefig(f"{self.output_dir}/{combo_name}_learning.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)

        fig3 = Visualizer.plot_confusion_matrix(cm_test, class1, class2, algorithm)
        fig3.savefig(f"{self.output_dir}/{combo_name}_confusion.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)

        result = {
            'algorithm': algorithm,
            'class1': class1,
            'class2': class2,
            'feature1': feature1,
            'feature2': feature2,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cm_test': cm_test,
            'epochs_run': len(model.errors_per_epoch if algorithm=='Perceptron' else model.mse_per_epoch),
            'converged': test_acc > 0.8
        }

        self.results.append(result)
        return result

    def generate_comprehensive_report(self):

        features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']

        combinations = [
            ('Adelie', 'Gentoo', 'CulmenLength', 'CulmenDepth'),
            ('Adelie', 'Gentoo', 'FlipperLength', 'BodyMass'),
            ('Chinstrap', 'Gentoo', 'CulmenLength', 'FlipperLength'),
            ('Adelie', 'Chinstrap', 'FlipperLength', 'BodyMass'),

            ('Adelie', 'Chinstrap', 'CulmenLength', 'CulmenDepth'),
            ('Adelie', 'Gentoo', 'OriginLocation', 'BodyMass'),
            ('Chinstrap', 'Gentoo', 'CulmenDepth', 'BodyMass'),

            ('Adelie', 'Chinstrap', 'CulmenLength', 'FlipperLength'),
            ('Adelie', 'Gentoo', 'CulmenDepth', 'FlipperLength'),
            ('Chinstrap', 'Gentoo', 'CulmenLength', 'BodyMass'),
        ]

        print("\n" + "="*70)
        print("PERCEPTRON ALGORITHM ANALYSIS")
        print("="*70)

        for combo in combinations:
            self.test_combination(*combo, algorithm='Perceptron', eta=0.01, epochs=100)

        print("\n" + "="*70)
        print("ADALINE ALGORITHM ANALYSIS")
        print("="*70)

        for combo in combinations:
            self.test_combination(*combo, algorithm='Adaline', eta=0.001, epochs=100, mse_threshold=0.5)

        self.generate_summary_report()

    def generate_summary_report(self):
        with open(f"{self.output_dir}/analysis_summary.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("PENGUIN CLASSIFICATION - ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("PERCEPTRON ALGORITHM RESULTS\n")
            f.write("-"*80 + "\n")
            perceptron_results = [r for r in self.results if r['algorithm'] == 'Perceptron']
            perceptron_results.sort(key=lambda x: x['test_accuracy'], reverse=True)

            for i, r in enumerate(perceptron_results, 1):
                f.write(f"\n{i}. {r['class1']} vs {r['class2']} | {r['feature1']} vs {r['feature2']}\n")
                f.write(f"   Train Accuracy: {r['train_accuracy']*100:.2f}%\n")
                f.write(f"   Test Accuracy: {r['test_accuracy']*100:.2f}%\n")
                f.write(f"   Epochs: {r['epochs_run']}\n")
                f.write(f"   Performance: {'GOOD' if r['test_accuracy'] > 0.85 else 'MODERATE' if r['test_accuracy'] > 0.7 else 'POOR'}\n")

            f.write("\n\n" + "="*80 + "\n")
            f.write("ADALINE ALGORITHM RESULTS\n")
            f.write("-"*80 + "\n")
            adaline_results = [r for r in self.results if r['algorithm'] == 'Adaline']
            adaline_results.sort(key=lambda x: x['test_accuracy'], reverse=True)

            for i, r in enumerate(adaline_results, 1):
                f.write(f"\n{i}. {r['class1']} vs {r['class2']} | {r['feature1']} vs {r['feature2']}\n")
                f.write(f"   Train Accuracy: {r['train_accuracy']*100:.2f}%\n")
                f.write(f"   Test Accuracy: {r['test_accuracy']*100:.2f}%\n")
                f.write(f"   Epochs: {r['epochs_run']}\n")
                f.write(f"   Performance: {'GOOD' if r['test_accuracy'] > 0.85 else 'MODERATE' if r['test_accuracy'] > 0.7 else 'POOR'}\n")

            f.write("\n\n" + "="*80 + "\n")
            f.write("BEST FEATURE COMBINATIONS (HIGHEST ACCURACY)\n")
            f.write("-"*80 + "\n")

            all_results = sorted(self.results, key=lambda x: x['test_accuracy'], reverse=True)
            for i, r in enumerate(all_results[:5], 1):
                f.write(f"\n{i}. {r['algorithm']}: {r['class1']} vs {r['class2']}\n")
                f.write(f"   Features: {r['feature1']} vs {r['feature2']}\n")
                f.write(f"   Test Accuracy: {r['test_accuracy']*100:.2f}%\n")

            f.write("\n\n" + "="*80 + "\n")
            f.write("KEY INSIGHTS AND ANALYSIS\n")
            f.write("-"*80 + "\n\n")

            f.write("1. FEATURE DISCRIMINATION:\n")
            f.write("   - FlipperLength and BodyMass show strong discriminative power\n")
            f.write("   - CulmenLength and CulmenDepth are effective for certain class pairs\n")
            f.write("   - OriginLocation provides geographical context\n\n")

            f.write("2. CLASS SEPARABILITY:\n")
            f.write("   - Gentoo vs Adelie: Generally highly separable\n")
            f.write("   - Gentoo vs Chinstrap: Good separability\n")
            f.write("   - Adelie vs Chinstrap: More challenging separation\n\n")

            f.write("3. ALGORITHM COMPARISON:\n")
            f.write("   - Perceptron: Fast convergence, binary classification\n")
            f.write("   - Adaline: Smooth learning, MSE-based optimization\n")
            f.write("   - Both achieve similar accuracy on linearly separable data\n\n")

            f.write("4. CONVERGENCE BEHAVIOR:\n")
            f.write("   - Well-separated classes converge in fewer epochs\n")
            f.write("   - Learning rate affects convergence speed\n")
            f.write("   - Some combinations require higher epochs for convergence\n\n")

        print(f"\nSummary report saved to: {self.output_dir}/analysis_summary.txt")


def main():
    print("Starting comprehensive analysis...")
    print("This will generate visualizations and analysis for multiple combinations.")
    print("Please wait...\n")

    generator = ReportGenerator()
    generator.generate_comprehensive_report()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {generator.output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
