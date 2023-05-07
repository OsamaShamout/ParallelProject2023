% % cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THRESHOLD .5
#define DATA_FILE "training_data100.txt"
#define LEARNING_RATE -.2
#define BIAS 0

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


float getThreshold(void);

/* Function checks the actual output of the perceptrons ouput
 * against the training data set. */
float checkOutput(int target, float actual_output);

void read_data(FILE *file_pointer, float *data, int data_points_count);

int count_data_points(FILE *file_pointer);

int NumberOfLines(FILE *file_pointer);

DataLine *readFile(FILE *file_pointer, int *num_lines);

/**********************/
/* SUM WEIGHTED INPUT */
/**********************/
/* A function for the dot product (summation) of weighted
 * inputs (i.e., (input1 * weight1) + (input2 * weight2) ... */
__device__ float sumWeightedInputs(float input1, float input2, float input3, float input4, float weight1, float weight2, float weight3, float weight4)
{
    /* sum means dot product here */
    float sum = 0;

    /* figure out the dot product here */
    sum = (input1 * weight1) + (input2 * weight2) + (input3 * weight3) + (input4 * weight4);
    sum = sum + BIAS; // add bias

    /* return sum */
    return sum;
}

/******************/
/* UPDATE WEIGHTS */
/******************/

/* This is the function that updates the weights
 * if the neuron misclassified input. */
/* I am using this formual to update weights:
 * new weight = old weight + (learning rate * current input * (error)
 * where error = expected output - actual output */
// __device__ inline float updateWeights(float weight, float learning_rate, float input, float error)
// {
//     float new_weight = 0;

//     /* use the formula here */
//     new_weight = weight + (learning_rate * input * error);

//     return new_weight;
// }

/***********************/
/* ACTIVATION FUNCTION */
/***********************/

/* Activation Function -- sees if the weights is greater
 * than a certain number, called threshold, and returns 1
 * and 0 therwise. */
__device__ int activationFunction(float dot_product, float threshold)
{
    if (dot_product >= threshold)
        return 0;
    /* object actual_outputified to class 0 */

    else
        return 1;
    /* object actual_outputified to class 1 */
}

/*****************/
/* CHECK OUTPUT  */
/*****************/

__global__ void train(DataLine *file, int num_lines, float *weight1, float *weight2, float *weight3, float *weight4, float threshold, float learning_rate, int *correct_predictions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_lines)
    {
        float input1 = file[i].input1;
        float input2 = file[i].input2;
        float input3 = file[i].input3;
        float input4 = file[i].input4;
        float target = file[i].target;

        float dot_product = sumWeightedInputs(input1, input2, input3, input4, *weight1, *weight2, *weight3, *weight4);
        int actual_output = activationFunction(dot_product, threshold);

        float error = target - actual_output;

        if (error != 0)
        {
            atomicAdd(weight1, learning_rate * error * input1);
            atomicAdd(weight2, learning_rate * error * input2);
            atomicAdd(weight3, learning_rate * error * input3);
            atomicAdd(weight4, learning_rate * error * input4);
        }
        else
        {
            atomicAdd(correct_predictions, 1);
        }
    }
}

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

    // Allocate memory on the GPU for the file data and copy it
    // Allocate memory on the GPU for the file data and copy it
    DataLine *d_file;
    cudaMalloc((void **)&d_file, num_lines * sizeof(DataLine));
    cudaMemcpy(d_file, file, num_lines * sizeof(DataLine), cudaMemcpyHostToDevice);

    // Allocate memory on the GPU for weights and copy their initial values
    float *d_weight1, *d_weight2, *d_weight3, *d_weight4;
    cudaMalloc((void **)&d_weight1, sizeof(float));
    cudaMalloc((void **)&d_weight2, sizeof(float));
    cudaMalloc((void **)&d_weight3, sizeof(float));
    cudaMalloc((void **)&d_weight4, sizeof(float));
    cudaMemcpy(d_weight1, &weight1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight2, &weight2, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight3, &weight3, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight4, &weight4, sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory on the GPU for correct_predictions and initialize it
    int *d_correct_predictions;
    cudaMalloc((void **)&d_correct_predictions, sizeof(int));
    cudaMemcpy(d_correct_predictions, &correct_predictions, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the CUDA kernel
    int blockSize = 256;
    int gridSize = (num_lines + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    train<<<gridSize, blockSize>>>(d_file, num_lines, d_weight1, d_weight2, d_weight3, d_weight4, threshold, learning_rate, d_correct_predictions);
    // Wait for the kernel to finish execution and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error executing the CUDA kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaEventRecord(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the results back from the GPU to the host
    cudaMemcpy(&weight1, d_weight1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&weight2, d_weight2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&weight3, d_weight3, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&weight4, d_weight4, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&correct_predictions, d_correct_predictions, sizeof(int), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_file);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_weight3);
    cudaFree(d_weight4);
    cudaFree(d_correct_predictions);

    printf("Final weights:\nw1: %.2f\nw2: %.2f\nw3: %.2f\nw4: %.2f\n", weight1, weight2, weight3, weight4);

    accuracy = (float)correct_predictions / (float)num_lines * 100.0;

    printf("Accuracy: %.2f%%\n", accuracy);
    // printf("Accuracy: %.2f%%\n", accuracy);
    printf("Kernel execution time: %f ms\n", elapsedTime);

    free(file);

    return 0;
}

/*************/
/* GET INPUT */
/*************/

/* gets a input for the perceptron from a data file */
float getInput(FILE *file_pointer, float *input1, float *input2, float *input3, float *input4, int *target)
{
    int n_read = fscanf(file_pointer, "%f %f %f %f %d", input1, input2, input3, input4, target);

    if (n_read != 5)
    {
        return 0;
    }
    return 1;
}

/******************/
/* GET THRESHOLD  */
/******************/
/* Ask the user for a threshold */
/* Note: for now I wont use this function.
 * Instead of asking the user for the threshold,
 * Ill just define it as a macro.
 */
float getThreshold(void)
{
    float threshold;

    printf("Threshold: ");
    scanf("%f", &threshold);

    return threshold;
}

/* Function checks the actual output of the perceptrons ouput against the expected output. */
float checkOutput(int target, float actual_output)
{
    float expected_output = target;

    /* error = expected_ouput - actual_output */
    float error = 0;
    /* the value of error is needed in the update
     * weight formula */

    printf("Expected Output: %.2f\n", expected_output);
    printf("Actual Output: %.2f\n", actual_output);

    /* calculate error */
    error = expected_output - actual_output;
    return error;
}
void read_data(FILE *file, float *data, int data_points_count)
{
    for (int i = 0; i < data_points_count; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            fscanf(file, "%f", &data[i * 5 + j]);
        }
    }

    fclose(file);
}

int count_data_points(FILE *file_pointer)
{
    int count = 0;
    float temp;
    while (fscanf(file_pointer, "%f", &temp) != EOF)
    {
        count++;
    }
    return count / 5;
}
DataLine *readFile(FILE *file_pointer, int *num_lines)
{
    DataLine *data = NULL;
    DataLine tmp;
    *num_lines = 0;

    while (fscanf(file_pointer, "%f %f %f %f %d", &tmp.input1, &tmp.input2, &tmp.input3, &tmp.input4, &tmp.target) == 5)
    {
        (*num_lines)++;
        data = (DataLine *)realloc(data, *num_lines * sizeof(DataLine));
        data[*num_lines - 1] = tmp;
    }

    return data;
}

int readLines(FILE *file_pointers)
{
    DataLine tmp;
    int num_lines = 0;

    while (fscanf(file_pointers, "%f %f %f %f %d", &tmp.input1, &tmp.input2, &tmp.input3, &tmp.input4, &tmp.target) == 5)
    {
        (num_lines)++;
    }

    return (num_lines);
}