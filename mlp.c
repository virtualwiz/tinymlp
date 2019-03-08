#include "mlp.h"

double sigmoid(double x){
  return 1 / (1 + (exp(-x)));
}

double dsigmoid(double x){
  return sigmoid(x) * (1 - sigmoid(x));
}

double error(int num_neurones, double* real_output, double* expected_output){
  double diff = 0;
  int i;
  for(i = 0; i < num_neurones; i++){
    diff += 0.5 * pow((*(real_output + i) - *(expected_output + i)), 2);
  }
  return diff;
}

void MLP_Dump(){
  int i,j;

  printf("Input layer:\t");
  for(i = 0; i < NUM_NEURONES_INPUT; i++){
    printf("%lf\t", neurone_input[i]);
  }
  printf("\n");
  for(i = 0; i < NUM_NEURONES_HIDDEN; i++){
    printf("W to hidden%d:\t",i);
    for(j = 0; j < NUM_NEURONES_INPUT; j++){
      printf("%lf\t", weight_i_h[j][i]);
    }
    printf("\n");
  }
  printf("Hidden layer:\t");
  for(i = 0; i < NUM_NEURONES_HIDDEN; i++){
    printf("%lf\t", neurone_hidden[i]);
  }
  printf("\n");
  for(i = 0; i < NUM_NEURONES_OUTPUT; i++){
    printf("W To output%d:\t",i);
    for(j = 0; j < NUM_NEURONES_HIDDEN; j++){
      printf("%lf\t", weight_h_o[j][i]);
    }
    printf("\n");
  }
  printf("Output layer:\t");
  for(i = 0; i < NUM_NEURONES_OUTPUT; i++){
    printf("%lf\t", neurone_output[i]);
  }
  printf("\n");

  printf("Output error:\t");
  printf("%lf", error(NUM_NEURONES_OUTPUT, neurone_output, neurone_input));
  printf("\n");
}

void MLP_Weights_Init(){
  int i;
  double rand_weight;
  time_t t;
  srand((unsigned) time(&t));
  for(i = 0; i < NUM_NEURONES_INPUT * NUM_NEURONES_HIDDEN; i++){
    rand_weight = (double)(rand() % 200) / 100 - 1;
    *(*(weight_i_h) + i) = rand_weight;
  }
  for(i = 0; i < NUM_NEURONES_HIDDEN * NUM_NEURONES_OUTPUT; i++){
    rand_weight = (double)(rand() % 200) / 100 - 1;
    *(*(weight_h_o) + i) = rand_weight;
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

void MLP_Train(int num_patterns, unsigned int num_epoches, double learning_rate, double** x, double** y){
  int i, k, i_epoch, i_pattern;
  double deltaj_output[NUM_NEURONES_OUTPUT];
  double deltaj_hidden[NUM_NEURONES_HIDDEN];
  double sumk_hidden;
  for(i_epoch = 1; i_epoch <= num_epoches; i_epoch++){
    for(i_pattern = 0; i_pattern < num_patterns; i_pattern++){
      /* Feed training vector */
      for(i = 0; i < NUM_NEURONES_INPUT; i++){
        neurone_input[i] = x[i_pattern][i];
      }
      /* Forward propagate */
      MLP_Evaluate();
      /* Compute delta for output layer */
      for(i = 0; i < NUM_NEURONES_OUTPUT; i++){
        deltaj_output[i] = (neurone_output[i] - y[i_pattern][i]) * neurone_output[i] * (1 - neurone_output[i]);
      }
      /* Compute delta for hidden layer */
      for(i = 0; i < NUM_NEURONES_HIDDEN; i++){
        for(k = 0; k < NUM_NEURONES_OUTPUT; k++){
          sumk_hidden += (weight_h_o[i][k] * deltaj_output[k]);
        }
        deltaj_hidden[i] = sumk_hidden * neurone_hidden[i] * (1 - neurone_hidden[i]);
      }
      /* Rebuild weight_i_h matrix */
      for(i = 0; i < NUM_NEURONES_INPUT; i++){
        for(k = 0; k < NUM_NEURONES_HIDDEN; k++){
          weight_i_h[i][k] -= learning_rate * deltaj_hidden[k] * neurone_input[i];
        }
      }
      /* Rebuild weight_h_o matrix */
      for(i = 0; i < NUM_NEURONES_HIDDEN; i++){
        for(k = 0; k < NUM_NEURONES_OUTPUT; k++){
          weight_h_o[i][k] -= learning_rate * deltaj_output[k] * neurone_hidden[i];
        }
      }
    }
  }
}

