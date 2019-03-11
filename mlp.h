#ifndef MATH_H
#define MATH_H
#include "math.h"
#endif

#ifndef STDLIB_H
#define STDLIB_H
#include "stdlib.h"
#endif

#ifndef STDIO_H
#define STDIO_H
#include "stdio.h"
#endif

#ifndef TIME_H
#define TIME_H
#include "time.h"
#endif

#ifndef MLP_H
#define MLP_H

/* Constants */
#define HIGH 1
#define SOFT_HIGH 0.9
#define LOW 0
#define SOFT_LOW 0.1

/* Define numbers of neurones in each layer.
   Excluding bias neurone. */
#define NUM_NEURONES_INPUT 4
#define NUM_NEURONES_HIDDEN 2
#define NUM_NEURONES_OUTPUT 4

/* Global variables used to define the neural network.
   TODO: Remove global variables. */
double neurone_input[NUM_NEURONES_INPUT + 1];
double neurone_hidden[NUM_NEURONES_HIDDEN + 1];
double neurone_output[NUM_NEURONES_OUTPUT];
double weight_i_h[NUM_NEURONES_INPUT + 1][NUM_NEURONES_HIDDEN + 1];
double weight_h_o[NUM_NEURONES_HIDDEN + 1][NUM_NEURONES_OUTPUT];

extern double MLP_ErrorAvg(int num_test_patterns, double* x, double* y);
extern void MLP_Dump(short int mode);
extern void MLP_Init();
extern void MLP_Evaluate();
extern void MLP_Train(int num_patterns, double learning_rate, double* x, double* y);

#endif

