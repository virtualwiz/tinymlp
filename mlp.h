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

#define HIGH 1
#define SOFT_HIGH 0.9
#define LOW 0
#define SOFT_LOW 0.1

#define NUM_NEURONES_INPUT 4
#define NUM_NEURONES_HIDDEN 2
#define NUM_NEURONES_OUTPUT 4

double neurone_input[NUM_NEURONES_INPUT];
double neurone_hidden[NUM_NEURONES_HIDDEN];
double neurone_output[NUM_NEURONES_OUTPUT];

double weight_i_h[NUM_NEURONES_INPUT][NUM_NEURONES_HIDDEN];
double weight_h_o[NUM_NEURONES_HIDDEN][NUM_NEURONES_OUTPUT];

extern double error(int num_neurones, double* real_output, double* expected_output);
extern void MLP_Dump();
extern void MLP_Weights_Init();
extern void MLP_Evaluate();
extern void MLP_Train(int num_patterns, unsigned int num_epoches, double learning_rate, double** x, double** y);

#endif
