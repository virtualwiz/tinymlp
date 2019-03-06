#ifndef MATH_H
#define MATH_H
#include "math.h"
#endif

#ifndef MLP_H
#define MLP_H

#define HIGH 1
#define HIGH_SOFT 0.9
#define LOW 0
#define LOW_SOFT 0.1

#define NUM_NEURONES_INPUT 4
#define NUM_NEURONES_HIDDEN 2
#define NUM_NEURONES_OUTPUT 4

double neurone_input[NUM_NEURONES_INPUT];
double neurone_hidden[NUM_NEURONES_HIDDEN];
double neurone_output[NUM_NEURONES_OUTPUT];

double weight_i_h[NUM_NEURONES_INPUT][NUM_NEURONES_HIDDEN];
double weight_h_o[NUM_NEURONES_HIDDEN][NUM_NEURONES_OUTPUT];

extern void MLP_Weights_Init();
extern void MLP_Evaluate();

#endif
