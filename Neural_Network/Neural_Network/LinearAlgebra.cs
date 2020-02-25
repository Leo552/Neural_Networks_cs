using System;

public static class LinearAlgebra
{
    public static double[,] mult2d_2d(double[,] matrix_1, double[,] matrix_2)
    {
        // Check that the matrix are the right dimensions
        if (!(matrix_1.GetLength(1) == matrix_2.GetLength(0)))
        {
            throw new ArgumentException("The matrix multiplication cannot be performed. The matrices are the wrong dimensions");
        }

        // It is going to run through and add numbers to a blank matrix
        int rows = matrix_1.GetLength(0);
        int columns = matrix_2.GetLength(1);
        double[,] output_mat = new double[rows, columns];
        double sum = 0;

        // Very compute intensive
        for (int c=0; c < columns; c++)
        {
            for (int r=0; r < rows; r++)
            {
                for (int i=0; i < columns; i++)
                {
                    sum += matrix_1[r, i] * matrix_2[i, c];
                }
                output_mat[r, c] = sum;
                sum = 0;
            }
        }

        // Return the output matrix
        return output_mat;
    }

    public static double[] mult2d_vec(double[,] matrix_1, double[] vec_2)
    {
        // Check that the matrix are the right dimensions
        if (!(matrix_1.GetLength(1) == vec_2.Length))
        {
            Console.WriteLine(Convert.ToString(matrix_1.GetLength(1)));
            Console.WriteLine(Convert.ToString(vec_2.Length));
            throw new ArgumentException("The matrix multiplication cannot be performed. The matrices are the wrong dimensions");
        }

        // It is going to run through and add numbers to a blank matrix
        int rows = matrix_1.GetLength(0);
        int columns = matrix_1.GetLength(1);
        double[] output_mat = new double[rows];
        double sum;

        // Very compute intensive
        for (int r = 0; r < rows; r++)
        {
            sum = 0;
            for (int i = 0; i < columns; i++)
            {
                sum += matrix_1[r, i] * vec_2[i];
            }
            output_mat[r] = sum;
        }
        
        // Return the output matrix
        return output_mat;
    }

    public static double[] Addvec_vec(double[] vec_1, double[] vec_2)
    {
        // Check that the matrix are the right dimensions
        if (!(vec_1.GetLength(0) == vec_2.GetLength(0)))
        {
            throw new ArgumentException("The vector addition cannot be performed. The vectors are the wrong dimensions");
        }

        // It is going to run through and add numbers to a blank matrix
        int rows = vec_1.GetLength(0);
        double[] output_mat = new double[rows];

        // Very compute intensive
        for (int r = 0; r < rows; r++)
        {
            output_mat[r] = vec_1[r] + vec_2[r];
        }

        // Return the output matrix
        return output_mat;
    }
}
