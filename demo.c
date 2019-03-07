#include "stdio.h"
#include "mlp.h"

int main(){

  unsigned int i;
  double optimal_err = 1;

  neurone_input[0] = 1;
  neurone_input[1] = 0;
  neurone_input[2] = 0;
  neurone_input[3] = 0;

  double optimal_weight_i_h[NUM_NEURONES_INPUT][NUM_NEURONES_HIDDEN];
  double optimal_weight_h_o[NUM_NEURONES_HIDDEN][NUM_NEURONES_OUTPUT];

  for(i = 0; i < 100000; i++){
    double err;
    MLP_Weights_Init();
    MLP_Evaluate();
    err = error(NUM_NEURONES_OUTPUT, neurone_output, neurone_input);
    if(err < optimal_err){
      optimal_err = err;
      for(i = 0; i < NUM_NEURONES_INPUT * NUM_NEURONES_HIDDEN; i++){
        *(*(optimal_weight_i_h) + i) = *(*(weight_i_h) + i);
      }
      for(i = 0; i < NUM_NEURONES_HIDDEN * NUM_NEURONES_OUTPUT; i++){
        *(*(optimal_weight_h_o) + i) = *(*(weight_h_o) + i);
      }
    }
    printf("Error is %lf after %d iterations.\n", optimal_err, i);
  }

  for(i = 0; i < NUM_NEURONES_INPUT * NUM_NEURONES_HIDDEN; i++){
    *(*(weight_i_h) + i) = *(*(optimal_weight_i_h) + i);
  }
  for(i = 0; i < NUM_NEURONES_HIDDEN * NUM_NEURONES_OUTPUT; i++){
    *(*(weight_h_o) + i) = *(*(optimal_weight_h_o) + i);
  }

  MLP_Evaluate();
  MLP_Dump();

  return 0;
}
