#include "mlp.h"

double sigmoid(double x){
  return 1 / (1 + (exp(-x)));
}

double dsigmoid(double x){
  return sigmoid(x) * (1 - sigmoid(x));
}

void MLP_Weights_Init(){
  int i;
  for(i = 0; i < NUM_NEURONES_INPUT * NUM_NEURONES_HIDDEN; i++){
    **(weight_i_h + i) = 1;
  }
  for(i = 0; i < NUM_NEURONES_HIDDEN * NUM_NEURONES_OUTPUT; i++){
    **(weight_h_o + i) = 1;
  }
}

void MLP_Evaluate(){
  int i_input, i_hidden, i_output;
  double sum_hidden[NUM_NEURONES_HIDDEN] = {0}, sum_output[NUM_NEURONES_OUTPUT] = {0};

  /* Evaluate sums, apply activation function and write to neurones */
  for(i_hidden = 0; i_hidden < NUM_NEURONES_HIDDEN; i_hidden++){
    for(i_input = 0; i_input < NUM_NEURONES_INPUT; i_input++){
      sum_hidden[i_hidden] += neurone_input[i_input] * weight_i_h[i_input][i_hidden];
    }
    neurone_hidden[i_hidden] = sigmoid(sum_hidden[i_hidden]);
  }

  for(i_output = 0; i_output < NUM_NEURONES_OUTPUT; i_output++){
    for(i_hidden = 0; i_hidden < NUM_NEURONES_HIDDEN; i_hidden++){
      sum_output[i_output] += neurone_hidden[i_hidden] * weight_h_o[i_hidden][i_output];
    }
    neurone_output[i_output] = sigmoid(sum_output[i_output]);
  }
}

