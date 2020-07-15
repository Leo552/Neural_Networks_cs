using System;
using System.Diagnostics;

namespace Neural_Network
{
    static class Program
    {
        static void Main(string[] args)
        {
            // Saved data properties
            string filename = "C:\\Users\\maxel\\OneDrive\\Machine learning\\Neural Networks\\Data_sets\\Boston_housing.csv";
            int[] X_columns = new int[13] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13 };
            int y_columns = 11;

            // Load the data
            (double[][] X, double[] y) = ML_tools.ReadCSV(filename, X_columns, y_columns);

            // Scale the data
            ML_tools.MinMaxTransformation_X(X);
            (double minimum, double range) = ML_tools.MinMaxTransformation_y(y);

            // Neural network parameters
            int hidden_layers = 2;
            int[] hidden_nodes = new int[2] { 5, 3 }; // Must be the same length as the hidden layers
            double l_rate_weights = 2;
            double l_rate_bias = 2;
            string activation_func = "sigmoid";

            NeuralNetwork Neural_Net = new NeuralNetwork(hidden_layers, hidden_nodes, l_rate_weights, l_rate_bias, 13, 1,activation_func);

            // Testing parameters
            int n_epoch = 10000;
            int k_folds = 5;

            // Time how long it takes to execute
            Stopwatch stopwatch = Stopwatch.StartNew();

            // Use cross validation to evaluate the algorithm
            Console.WriteLine("Cross validation absolute error: {0}", Neural_Net.CrossValidationEvaluation(X, y, k_folds, n_epoch));
            stopwatch.Stop();
            Console.WriteLine("Execution time: {0} seconds", (stopwatch.ElapsedMilliseconds/1000));
            Console.ReadLine();
        }
    }
}
