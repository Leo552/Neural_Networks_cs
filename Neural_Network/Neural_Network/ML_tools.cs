using System;
using System.Collections.Generic;
using System.IO;

public static class ML_tools
{
    public static (double[][], double[]) ReadCSV(string filename, int[] X_columns, int y_column, string action_missing_vals = "remove_example")
    {
        List<double[]> list_X = new List<double[]>();
        List<double> list_y = new List<double>();
        bool missing_vals;

        using (var reader = new StreamReader(@filename))
        {
            List<double> list_temp = new List<double>();

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');
                missing_vals = false;
                list_temp.Clear();

                foreach (int i in X_columns)
                {
                    try
                    {
                        // This may fail if there is missing data
                        list_temp.Add(Convert.ToDouble(values[i]));
                    }
                    catch
                    {
                        missing_vals = true;
                    }
                }

                if (missing_vals && action_missing_vals == "remove_example")
                {
                    // Don't include this example in the dataset
                    continue;
                }
                else
                {
                    list_X.Add(list_temp.ToArray());
                    list_y.Add(Convert.ToDouble(values[y_column]));
                }
            }
        }

        return (list_X.ToArray(), list_y.ToArray());
    }

    public static (int[], int) CrossValidationSplit(double[][] X, int k_folds)
    {
        int[] dataset_split = new int[k_folds];
        int fold_size = Convert.ToInt32(Math.Floor(Convert.ToDouble(X.Length / k_folds)));

        for (int k = 0; k < k_folds; k++)
        {
            dataset_split[k] = k * fold_size;
        }

        return (dataset_split, fold_size);
    }

    public static void MinMaxTransformation_X(double[][] X)
    {
        int no_examples = X.Length;
        int no_features = X[0].Length;

        double[] maximum = new double[no_features];
        double[] minimum = new double[no_features];
        double[] range = new double[no_features];

        // Fill these with zeros
        Populate<double>(maximum, 0);
        Populate<double>(minimum, 10000000);

        // Find the maximum and minimum
        for (int i = 0; i < no_examples; i++)
        {
            for (int j = 0; j < no_features; j++)
            {
                if (X[i][j] > maximum[j])
                {
                    maximum[j] = X[i][j];
                }
                else if (X[i][j] < minimum[j])
                {
                    minimum[j] = X[i][j];
                }
            }
        }

        // Calculate the range
        for (int j = 0; j < no_features; j++)
        {
            range[j] = maximum[j] - minimum[j];
        }

        // Implement the transformation
        for (int i = 0; i < no_examples; i++)
        {
            for (int j = 0; j < no_features; j++)
            {
                X[i][j] = (X[i][j] - minimum[j]) / range[j] + 0.01; // Can't be zero #######################
                if (X[i][j] <= 0)
                {
                    Console.WriteLine(X[i][j]);
                }
            }
        }
    }

    public static (double, double) MinMaxTransformation_y(double[] y)
    {
        int no_examples = y.Length;

        double maximum = 0;
        double minimum = 100000000;
        double range;

        // Find the maximum and minimum
        for (int i = 0; i < no_examples; i++)
        {

            if (y[i] > maximum)
            {
                maximum = y[i];
            }
            else if (y[i] < minimum)
            {
                minimum = y[i];
            }

        }

        // Calculate the range
        range = maximum - minimum;

        // Implement the transformation
        for (int i = 0; i < no_examples; i++)
        {
            y[i] = (y[i] - minimum) / range;
        }

        return (minimum, range);
    }

    public static void Populate<T>(this T[] arr, T value)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = value;
        }
    }

    // Taken from https://www.dotnetperls.com/array-slice
    public static T[] Slice<T>(this T[] source, int start, int end)
    {
        // Handles negative ends.
        if (end < 0)
        {
            end = source.Length + end;
        }
        int len = end - start;

        // Return new array.
        T[] res = new T[len];
        for (int i = 0; i < len; i++)
        {
            res[i] = source[i + start];
        }
        return res;
    }
}
