#include <stdio.h>
#include <stdlib.h>
#define DATA_FILE "training_data.txt"

typedef struct
{
    float input1;
    float input2;
    float input3;
    float input4;
    int target;
} DataLine;

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

int *readLines(FILE *file_pointers)
{
    DataLine *data = NULL;
    DataLine tmp;
    int num_lines = 0;

    while (fscanf(file_pointers, "%f %f %f %f %d", &tmp.input1, &tmp.input2, &tmp.input3, &tmp.input4, &tmp.target) == 5)
    {
        (num_lines)++;
    }

    return (num_lines);
}

int main(){

    int num_lines = 0;

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
    DataLine *file = readFile(file_pointer1, &num);
    fclose(file_pointer1);

    printf("Number of lines: %d\n", num_lines);
    printf("File contents are:\n");
    for (int i = 0; i < num_lines; i++)
    {
        printf("%.2f %.2f %.2f %.2f %d\n", file[i].input1, file[i].input2, file[i].input3, file[i].input4, file[i].target);
    }

    // Free the memory allocated for the data array
    free(file);
    return 0;
}