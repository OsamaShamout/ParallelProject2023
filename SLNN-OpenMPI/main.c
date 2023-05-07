/**********************************/
/* How the data file works:       */
/* ^Each line is a new set        */
/* ^the first number is input1    */
/* ^the second number is input2   */
/* ^the third number is input3    */
/* ^the fourth number is the class*/
/*                                */
/**********************************/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include "functions.h"
#include <mpi.h>

/* all function prototypes are here */

/* _______________ */
/*                 */
/*     MACROS      */
/* _______________ */
#define THRESHOLD .5
#define DATA_FILE "training_data.txt"
#define LEARNING_RATE -.2
// NOTE: bias defined in functions.c

int main(int argc, char *argv[])
{
    int correct_predictions = 0;
    int total_predictions = 0;
    float accuracy = 0;
    double start_comm_time, end_comm_time, total_comm_time = 0;
    double start_comp_time, end_comp_time, total_comp_time = 0;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("PERCEPTRON TRAINING ALGORITHM IMPLEMENTATION\n");
    }
    DataLine *file = NULL;
    int num_lines = 0;
    if (rank == 0)
    {
        num_lines = 0;

        FILE *file_pointer = fopen(DATA_FILE, "r");

        if (file_pointer == NULL)
        {
            printf("Error: Unable to open the data file.\n");
            exit(1);
        }

        num_lines = readLines(file_pointer);
        fclose(file_pointer);

        FILE *file_pointer1 = fopen(DATA_FILE, "r");

        if (file_pointer1 == NULL)
        {
            printf("Error: Unable to open the data file.\n");
            exit(1);
        }
        int num = 0;
        file = readFile(file_pointer1, &num);
        fclose(file_pointer1);

        printf("Number of lines: %d\n", num_lines);
        printf("File contents are:\n");
        for (int i = 0; i < num_lines; i++)
        {
            printf("%.2f %.2f %.2f %.2f %d\n", file[i].input1, file[i].input2, file[i].input3, file[i].input4, file[i].target);
        }
    }
    start_comm_time = MPI_Wtime();
    // Broadcast the number of data points to all processes
    MPI_Bcast(&num_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        file = (DataLine *)malloc(num_lines * sizeof(DataLine));
    }

    // Divide the dataset among processes
    int data_per_process = num_lines / size;
    DataLine *local_data = (DataLine *)malloc(data_per_process * sizeof(DataLine));

    // Calculate the offset for each process
    int offset = rank * data_per_process * 5;

    for (int i = 0; i < data_per_process; i++)
    {
        local_data[i] = file[offset + i];
    }
    start_comm_time = MPI_Wtime();
    // Scatter the dataset to all processes
    MPI_Scatter(file, data_per_process * 5, MPI_FLOAT, local_data, data_per_process * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);
    end_comm_time = MPI_Wtime();
    total_comm_time += (end_comm_time - start_comm_time);
    MPI_Barrier(MPI_COMM_WORLD);
    /* ___________________ */
    /*                     */
    /*     VARIABLES       */
    /* ___________________ */

    float input1 = 0, input2 = 0, input3 = 0, input4 = 0;
    /* the inputs for the artifical neural network */

    float weight1 = 0, weight2 = 0, weight3 = 0, weight4 = 0;
    /* the weights for the artifical neural network */

    float threshold = 0;
    /* used in Activation Function */
    /* if summation of weighted inputs >= threshold,
     * Activation Function returns true. Otherwise,
     * Activation Function returns false. */

    float learning_rate = LEARNING_RATE;

    float dot_product = 0;
    /* dot product = (a1 * b1) + (a2 * b2) + ... + (an * bn) */
    /* This will be the summation of all the weighted inputs.
     * This value will be given to the activation function. */

    /* What actual_output is the object being classified in? */
    int actual_output = 0;

    /* error = expected output - actual ouput */
    /* error is used in the update weight formula */
    float error = 0;

    /* I will keep track of if there are any incorrect
     * classifications left by using a boolean value.
     * true means there are still incorrect classifications
     * flase means all classifications are correct. */
    bool incorrectClassifications = true;
    /* ___________________ */
    /*                     */
    /*        INPUT        */
    /* ___________________ */

    /* We need to seed the random number generator.
     * Otherwise, it will produce the same number every time
     * the program is run. */
    srand(time(NULL));
    /* I am using the current time to seed rand(). */

    /* get threshold from the user */
    // threshold = getThreshold();

    /* get threshold from macro */
    threshold = THRESHOLD;

    /* The weights will start off as random numbers in
     * the range [0, 1]. */
    weight1 = ((float)rand() / (float)(RAND_MAX / 1.0));
    weight2 = ((float)rand() / (float)(RAND_MAX / 1.0));
    weight3 = ((float)rand() / (float)(RAND_MAX / 1.0));
    weight4 = ((float)rand() / (float)(RAND_MAX / 1.0));

    // int local_correct_predictions = 0;
    // int local_total_predictions = 0;
    start_comp_time = MPI_Wtime();
    while (incorrectClassifications == true)
    {
        incorrectClassifications = false;

        /* get input from the data set file */

        for (int i = 0; i < data_per_process; i++)
        {
            input1 = local_data[i].input1;
            input2 = local_data[i].input2;
            input3 = local_data[i].input3;
            input4 = local_data[i].input4;
            float target = local_data[i].target;

            /* ___________________ */
            /*                     */
            /*     CALCULATION     */
            /* ___________________ */

            /* sum the weighted inputs */
            dot_product = sumWeightedInputs(input1, input2, input3, input4, weight1, weight2, weight3, weight4);

            /* apply activation function to sum of weighted inputs */
            actual_output = activationFunction(dot_product, threshold);

            /* ___________________ */
            /*                     */
            /*       OUPUT         */
            /* ___________________ */

            /* print which data set we are on */
            printf("Data Set %d\n", i);

            /* print the inputs */
            printf("\n"); // new line
            printf("Input 1 = %.2f\n", input1);
            printf("Input 2 = %.2f\n", input2);
            printf("Input 3 = %.2f\n", input3);
            printf("Input 4 = %.2f\n", input4);

            /* print the weights */
            printf("\n"); // new line
            printf("Weight 1 = %.2f\n", weight1);
            printf("Weight 2 = %.2f\n", weight2);
            printf("Weight 3 = %.2f\n", weight3);
            printf("Weight 4 = %.2f\n", weight4);

            /* print the summation of weighted inputs */
            printf("\n"); // new line
            printf("Summation = %.2f\n", dot_product);

            /* print the actual_output */
            printf("Object classified to class %d.\n", actual_output);

            /* check the output */
            error = checkOutput(target, actual_output);

            // local_total_predictions++;
            /* print the result */
            if (error == 0)
            {
                printf("Ouput correct.\n");
                // local_correct_predictions++;
            }
            else
            {
                /* set incorrectClassifications to true
                 * to loop through the data set once more */
                incorrectClassifications = true;

                printf("Output incorrect.\n");
                printf("Error = %.0f\n", error);

                /* we need to update the weights if *
                 * there is an error */
                weight1 = updateWeights(weight1, learning_rate, input1, error);
                weight2 = updateWeights(weight2, learning_rate, input2, error);
                weight3 = updateWeights(weight3, learning_rate, input3, error);
                weight4 = updateWeights(weight4, learning_rate, input4, error);

                /* print the new weights */
                printf("\n"); // new line;
                printf("NEW WEIGHTS: \n");
                printf("*** New weight 1: %.2f\n", weight1);
                printf("*** New weight 2: %.2f\n", weight2);
                printf("*** New weight 3: %.2f\n", weight3);
                printf("*** New weight 4: %.2f\n", weight4);
            } // ends else

            printf("\n");
            printf("-----------------------------------------------\n");
            printf("-----------------------------------------------\n\n");


        } // ends while (1)
    }
    // start_comm_time = MPI_Wtime();
    // MPI_Reduce(&local_correct_predictions, &correct_predictions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&local_total_predictions, &total_predictions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // end_comm_time = MPI_Wtime();
    // total_comm_time += (end_comm_time - start_comm_time);
    // end_comp_time = MPI_Wtime();
    // total_comp_time += (end_comp_time - start_comp_time);
    float *all_weights = NULL;

    if (rank == 0)
    {
        all_weights = (float *)malloc(size * 4 * sizeof(float));
    }

    float local_weights[] = {weight1, weight2, weight3, weight4};
    start_comm_time = MPI_Wtime();
    MPI_Gather(local_weights, 4, MPI_FLOAT, all_weights, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
    end_comm_time = MPI_Wtime();
    total_comm_time += (end_comm_time - start_comm_time);
    if (rank == 0)
    {
        float avg_weight1 = 0, avg_weight2 = 0, avg_weight3 = 0, avg_weight4 = 0;

        for (int i = 0; i < size * 4; i += 4)
        {
            avg_weight1 += all_weights[i];
            avg_weight2 += all_weights[i + 1];
            avg_weight3 += all_weights[i + 2];
            avg_weight4 += all_weights[i + 3];
        }

        avg_weight1 /= size;
        avg_weight2 /= size;
        avg_weight3 /= size;
        avg_weight4 /= size;

        printf("Final Model Weights:\nWeight 1: %f\nWeight 2: %f\nWeight 3: %f\nWeight 4: %f\n",
               avg_weight1, avg_weight2, avg_weight3, avg_weight4);

        free(all_weights);
    }

    start_comm_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    end_comm_time = MPI_Wtime();
    total_comm_time += (end_comm_time - start_comm_time);

    if (rank == 0)
    {
        // accuracy = (float)correct_predictions / (float)total_predictions * 100;
        // printf("Accuracy: %.2f%%\n", accuracy);
        printf("Total time: %f\n", total_comp_time + total_comm_time);
        // printf("Total communication time: %f\n", total_comm_time);
        // printf("Total computation time: %f\n", total_comp_time);
        free(file);
    }
    free(local_data);
    

    // Finalize MPI
    MPI_Finalize();

    return 0;
    // end//
}
