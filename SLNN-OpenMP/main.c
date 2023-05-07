#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <omp.h>
#include "functions.h"

#define THRESHOLD .5
#define DATA_FILE "training_data.txt"
#define LEARNING_RATE -.2

int main(int argc, char *argv[])
{
    int correct_predictions = 0;
    float accuracy = 0;
    double start_time, end_time;

    printf("PERCEPTRON TRAINING ALGORITHM IMPLEMENTATION\n");

    DataLine *file = NULL;
    int num_lines = 0;

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

    int num_threads = 1;
    printf("Using %d threads\n", num_threads);

    start_time = omp_get_wtime();
    // Parallelize the training loop
    float input1 = 0, input2 = 0, input3 = 0, input4 = 0;
    float weight1 = 0, weight2 = 0, weight3 = 0, weight4 = 0;
    float threshold = THRESHOLD;
    float learning_rate = LEARNING_RATE;
    float dot_product = 0;
    int actual_output = 0;
    float error = 0;

    weight1 = ((float)rand() / (float)(RAND_MAX / 1.0));
    weight2 = ((float)rand() / (float)(RAND_MAX / 1.0));
    weight3 = ((float)rand() / (float)(RAND_MAX / 1.0));
    weight4 = ((float)rand() / (float)(RAND_MAX / 1.0));

#pragma omp parallel for private(input1, input2, input3, input4, dot_product, actual_output, error) reduction(+ : weight1, weight2, weight3, weight4)
        for (int i = 0; i < num_lines; i++)
        {
            input1 = file[i].input1;
            input2 = file[i].input2;
            input3 = file[i].input3;
            input4 = file[i].input4;
            float target = file[i].target;

            dot_product = sumWeightedInputs(input1, input2, input3, input4, weight1, weight2, weight3, weight4);
            actual_output = activationFunction(dot_product, threshold);

            error = target - actual_output;

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

            if (error != 0)
            {
                weight1 += learning_rate * error * input1;
                weight2 += learning_rate * error * input2;
                weight3 += learning_rate * error * input3;
                weight4 += learning_rate * error * input4;

                printf("\n"); // new line;
                printf("NEW WEIGHTS: \n");
                printf("*** New weight 1: %.2f\n", weight1);
                printf("*** New weight 2: %.2f\n", weight2);
                printf("*** New weight 3: %.2f\n", weight3);
                printf("*** New weight 4: %.2f\n", weight4);
            }
        }
            
//             else
//             {
// #pragma atomic
//                 correct_predictions++;
//             }
//         }
//     }

    printf("Final weights:\nw1: %.2f\nw2: %.2f\nw3: %.2f\nw4: %.2f\n", weight1, weight2, weight3, weight4);
    // end of omp parallel
    end_time = omp_get_wtime();

    // accuracy = (float)correct_predictions / (float)num_lines * 100.0;
    // printf("Accuracy: %.2f%%\n", accuracy);
    printf("Elapsed time: %.4f seconds\n", end_time - start_time);

    free(file);

    return 0;
}
