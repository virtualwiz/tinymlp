#include "stdio.h"
#include "mlp.h"

#define NUM_PATTERNS 4

int main(){
  double pattern_set[NUM_PATTERNS * NUM_NEURONES_INPUT] = {
    SOFT_HIGH, SOFT_LOW, SOFT_LOW, SOFT_LOW,
    SOFT_LOW, SOFT_HIGH, SOFT_LOW, SOFT_LOW,
    SOFT_LOW, SOFT_LOW, SOFT_HIGH, SOFT_LOW,
    SOFT_LOW, SOFT_LOW, SOFT_LOW, SOFT_HIGH,
  };
  int i, j;

  MLP_Weights_Init();
  MLP_Dump();

  MLP_Train(NUM_PATTERNS, 1000, 0.0002, pattern_set, pattern_set);

  for(j = 0; j < NUM_PATTERNS; j++){
    for(i = 0; i < NUM_NEURONES_INPUT; i++){
      neurone_input[i] = pattern_set[NUM_NEURONES_INPUT * j + i];
    }
    MLP_Evaluate();
    MLP_Dump();
  }

  return 0;
}
