#include <stdio.h>
/* for FILE type */

typedef struct
{
    float input1;
    float input2;
    float input3;
    float input4;
    int target;
} DataLine;

int readLines(FILE *file_pointer);

/* get inputs for the perceptron from a data file */
float getInput(FILE *file_pointer, float *input1, float *input2, float *input3, float *input4, int *target);

/* Ask the user for a threshold */
/* Note: for now I won't use this function, I'll just
 * use a macro for the threshold */
float getThreshold(void);

/* A function for the dot product (summation) of weighted
 * inputs (i.e., (input1 * weight1) + (input2 * weight2) ... */
float sumWeightedInputs(float input1, float input2, float input3, float input4, float weight1, float weight2, float weight3, float weight4);

/* Activation Function -- sees if the weights is greater
 * than a certain number, called threshold, and returns 1
 * and 2 otherwise */
int activationFunction(float dot_product, float threshold);

/* Function checks the actual output of the perceptron's ouput
 * against the training data set. */
float checkOutput(int target, float actual_output);

/* a function to update weights */
float updateWeights(float weight, float learning_rate, float input, float error);

void read_data(FILE *file_pointer, float *data, int data_points_count);

int count_data_points(FILE *file_pointer);

int NumberOfLines(FILE *file_pointer);

DataLine *readFile(FILE *file_pointer, int *num_lines);