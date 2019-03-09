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

  MLP_Weights_Init();
  MLP_Train(NUM_PATTERNS, 3000, 0.0002, pattern_set, pattern_set);
  MLP_Dump();

  return 0;
}
