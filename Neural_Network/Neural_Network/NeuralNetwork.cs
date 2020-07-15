using System;
using System.Collections.Generic;

public class NeuralNetwork
{
    // Help from: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

    public int no_layers;
    public double l_rate_weights;
    public double l_rate_bias;
    public double[][,] all_weights;
    public double[][] all_bias;
    public double[][] neurons;
    public double[][] partial_gradient_vector;

    public NeuralNetwork(int hidden_layers, int[] hidden_nodes, double l_rate_weights, double l_rate_bias, int no_features, int output_nodes = 1, string activation_func = "sigmoid")
    {
        // Initiate the graph network
        no_layers = hidden_layers + 2; // Include the input and output layers
        (this.all_weights, this.all_bias, this.neurons, this.partial_gradient_vector)  = GraphNetwork.Initiate_GraphNetwork(hidden_nodes, no_layers, no_features, output_nodes, 123);

        // Assign learning rate
        this.l_rate_weights = l_rate_weights;
        this.l_rate_bias = l_rate_bias;
        
        // Assign the activation function - ############ Sort this out
    }

    // Train the network
    public void Train(int n_epoch, double[][] X, double[] y, bool print_vals = false)
        // Currently running an online learning setup
    {
        // Check the dimensions are right
        if (!(X.Length == y.Length))
        { throw new ArgumentException("The X and y training data must have the same number of examples"); }

        for (int epoch = 0; epoch < n_epoch; epoch++)
        {
            double sum_squared_error = 0;
            double prediction;

            // Online example by example mode
            for (int i = 0; i < X.GetLength(0); i++)
            {
                // Forward propagate - fill the neurons with the 
                prediction = FeedForward(X[i]);

                // Calculate the error this setup produces
                sum_squared_error += Math.Pow((prediction - y[i]), 2);

                // Back propagate
                double[] expected = new double[1] { y[i] }; // For multiple output nodes
                BackPropagation(expected);

                // Update the weights
                UpdateWeights(X[i]);
            }

            if (print_vals)
            {
                Console.WriteLine(Convert.ToString(sum_squared_error / X.Length));
                Console.WriteLine(Convert.ToString(epoch));
            }
        }
    }

    // Evaluate the algorithm using cross validation and return average score
    public double CrossValidationEvaluation(double[][] X, double[] y, int k_folds, int n_epoch)
    {
        // Get the k fold validation split points
        int[] dataset_split = new int[k_folds];
        int fold_size;
        (dataset_split, fold_size) = ML_tools.CrossValidationSplit(X, k_folds);

        // Set up some more parameters
        double[][] train_X;
        double[] train_y;
        double[][] test_X;
        double[] test_y;

        double score = 0;
        double[] predictions = new double[fold_size];

        // Run throught the algorithm k times
        for (int k = 0; k < k_folds - 1; k++)
        {
            // Bit fiddly here to get the right data in the training set
            if (!(k == 0))
            {
                List<double[]> list_X = new List<double[]>();
                list_X.AddRange(ML_tools.Slice(X, dataset_split[0], dataset_split[k] - 1));
                List<double> list_y = new List<double>();
                list_y.AddRange(ML_tools.Slice(y, dataset_split[0], dataset_split[k] - 1));

                if (!(k == k_folds - 1 || k == k_folds - 2))
                {
                    list_X.AddRange(ML_tools.Slice(X, dataset_split[k + 1], dataset_split[k + 2] - 1));
                    list_y.AddRange(ML_tools.Slice(y, dataset_split[k + 1], dataset_split[k + 2] - 1));
                }
                train_X = list_X.ToArray();
                train_y = list_y.ToArray();

            }
            else
            {
                train_X = ML_tools.Slice(X, dataset_split[k + 1], dataset_split[k + 2] - 1);
                train_y = ML_tools.Slice(y, dataset_split[k + 1], dataset_split[k + 2] - 1);
            }
            test_X = ML_tools.Slice(X, dataset_split[k], dataset_split[k + 1] - 1);
            test_y = ML_tools.Slice(y, dataset_split[k], dataset_split[k + 1] - 1);

            // Train the network
            Train(n_epoch, train_X, train_y);

            // Make predictions with it
            predictions = Predict(test_X);

            // Find the ABSE and average it
            score = ((score * k) + AbsoluteError(predictions, test_y)) / (k + 1);
        }
        return score;
    }

    // Predict the result for a given list of examples
    public double[] Predict(double[][] input_X)
    {
        double[] predictions = new double[input_X.Length];

        for (int i = 0; i < input_X.Length; i++)
        {
            predictions[i] = FeedForward(input_X[i]);
        }
        return predictions;
    }

    // Update the weights and bias'
    private void UpdateWeights(double[] example)
    {
        double[] inputs;
        for (int layer = 0; layer < no_layers - 2; layer++)
        {
            if (layer == 0)
            {
                inputs = example;
            }
            else
            {
                inputs = neurons[layer];
            }
            for(int i = 0; i < neurons[layer + 1].Length; i++)
            {
                for(int j = 0; j < inputs.Length - 1; j++)
                {
                    all_weights[layer][i, j] += l_rate_weights * partial_gradient_vector[layer][i] * inputs[j];
                }
                all_bias[layer][i] += l_rate_bias * partial_gradient_vector[layer][i];
            }
        }
    }

    // Returns a prediction with the current weights and bias'
    private double FeedForward(double[] example)
    {
        double prediction;

        // Index of 0 indicates the input layer
        neurons[0] = example;

        for (int layer = 0; layer < no_layers - 1; layer++)
        {
            // Mulitiply by the weights then add the bias
            neurons[layer + 1] = LinearAlgebra.mult2d_vec(all_weights[layer], neurons[layer]);
            neurons[layer + 1] = LinearAlgebra.Addvec_vec(neurons[layer + 1], all_bias[layer]);

            // Don't apply the activation function to the last layer output neurons
            if (!(layer + 1 == no_layers - 1))
            {
                Sigmoid(ref neurons[layer + 1]);
            }
        }
        prediction = neurons[no_layers - 1][0]; // 0 Since there is only one ouput node

        return prediction;
    }

    // Application of the chain rule to find the derivative of the cost function slope - Stochastic Gradient Descent
    private void BackPropagation(double[] expected_val) 
    {
        // Go backwards through the layers
        for (int layer = no_layers - 1; layer > 0; layer--)
        {
            double[] errors = new double[neurons[layer].GetLength(0)];

            if (!(layer == no_layers - 1))
            {
                for (int node = 0; node < neurons[layer].GetLength(0); node++)
                {
                    double error = 0;
                    for (int node_next_layer = 0; node_next_layer < neurons[layer + 1].GetLength(0); node_next_layer++)
                    {
                        error += all_weights[layer][node_next_layer, node] * partial_gradient_vector[layer][node_next_layer];
                    }
                    errors[node] = error;
                }
            }
            else
            {
                for(int i = 0; i < neurons[layer].Length; i++) // This is usually a 1 node output
                {
                    errors[i] = expected_val[i] - neurons[layer][i];
                }
            }

            for (int i = 0; i < neurons[layer].Length; i++)
            {
                // The sigmoid function is not applied to the last node
                if(!(layer == no_layers - 1))
                {
//                    Console.WriteLine(Sigmoid_derivative(neurons[layer][i]));
//                    Console.WriteLine(errors[i] * Sigmoid_derivative(neurons[layer][i]));
                    partial_gradient_vector[layer - 1][i] = errors[i] * Sigmoid_derivative(neurons[layer][i]);
                }
                else
                {
                    partial_gradient_vector[layer - 1][i] = errors[i] * neurons[layer][i];
                }
            }
        }
    }

    // An activation function for the neural network - modifies a running vector of values for the nodes
    public static void Sigmoid(ref double[] run_vec)
    {
        double k;
        for (int i = 0; i < run_vec.GetLength(0); i++)
        {
            k = Math.Exp(run_vec[i]);
            run_vec[i] = k / (1.0f + k);
        }
    }

    // Sigmoid derivative
    public static double Sigmoid_derivative(double output)
    {
        return output * (1.0 - output); // See note for derivation
    }

    // Get a single array from the row of a 2d array
    public static double[] SliceRow(double[,] array, int row)
    {
        double[] out_array = new double[array.GetLength(1)];
        for (int i = 0; i < array.GetLength(1); i++)
        { out_array[i] = array[row, i]; }

        return out_array;
        
    }

    // Returns the MSE of the predictions
    private double MeanSquaredError(double[] predictions, double[] expected)
    {
        double summed_squares = 0;
        int num_examples = predictions.GetLength(0);
        for (int i = 0; i < num_examples; i++)
        {
            summed_squares += Math.Pow(predictions[i] - expected[i], 2);
        }
        return (summed_squares / num_examples);
    }

    // Returns the ABSE of the predictions
    private double AbsoluteError(double[] predictions, double[] expected)
    {
        double summed_squares = 0;
        int num_examples = predictions.GetLength(0);
        for (int i = 0; i < num_examples; i++)
        {
            summed_squares += Math.Abs(predictions[i] - expected[i]);
        }
        return (summed_squares / num_examples);
    }
}