using System;

// This classs is specific to feed forward networks i.e. every node connects to every other node in the following layer
public static class GraphNetwork
{
    public static (double[][,], double[][], double[][], double[][]) Initiate_GraphNetwork(int[] graph, int no_layers, int no_features, int output_nodes = 1, UInt16 seed = 123)
    // Graph is an array of the number of nodes in each layer
    // no_layers pass the number of layers
    {
        // Add the input and output layers to the graph
        AddInputOutputNodes(ref graph, no_features, output_nodes);

        // Create a 1d array (layers) of 2d arrays (weights) to represent the node weightings
        double[][] neurons = new double[no_layers][]; // Will contain the activations from each neuron
        double[][] partial_gradient_vector = new double[no_layers - 1][];
        double[][,] all_weights = new double[no_layers - 1][,];
        double[][] all_bias = new double[no_layers - 1][];
        int start_bias = 0;
        double previous_result = seed;

        for (int layer = 0; layer < no_layers; layer++) // -1 since there must be at least two layers to have one set of weights
        {
            neurons[layer] = new double[graph[layer]];

            if (!(layer == 0))
            {
                partial_gradient_vector[layer - 1] = new double[graph[layer]];
                all_weights[layer - 1] = new double[graph[layer], graph[layer - 1]];
                all_bias[layer - 1] = new double[graph[layer]];

                for (int row = 0; row < graph[layer]; row++) // Number of nodes in this layer
                {
                    for (int col = 0; col < graph[layer - 1]; col++) // The number of nodes in the previous layer
                    {
                        previous_result = LCG_rand(previous_result);
                        all_weights[layer - 1][row, col] = (LCG_rand(previous_result) / 1000); // Make the psuedo random number a lot smaller

                        // Why you must initiate with random weights
                        // https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
                    }

                    all_bias[layer - 1][row] = start_bias;
                }
            }
        }

        return (all_weights, all_bias, neurons, partial_gradient_vector);
    }
    
    static private void AddInputOutputNodes(ref int[] graph, int no_features, int output_nodes)
    {
        Array.Resize(ref graph, graph.Length + 2);
        for(int i = graph.Length - 1; i > -1; i--)
        {
            if (i == 0) { graph[i] = no_features; }
            else if ( i == graph.Length - 1) { graph[i] = output_nodes; }
            else { graph[i] = graph[i - 1]; }

        }
    }

    static private double LCG_rand(double X_n)
    // The Linear Congruential Generator
    // https://www.geeksforgeeks.org/pseudo-random-number-generator-prng/
    {
        // Seed - this needs changing regularly
        UInt16 m = 1024; // Modulus 0 < m
        UInt16 a = 243;  // Mulitplier 0 < a < m - can't exceed  2^16/m
        UInt16 c = 113; // Increment 0 < c < m

        return (a * X_n + c) % m;
    }
}
