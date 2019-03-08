#include "stdio.h"
#include "mlp.h"

int main(){
  double pattern_set[4][4] = {
    {SOFT_HIGH, SOFT_LOW, SOFT_LOW, SOFT_LOW},
    {SOFT_LOW, SOFT_HIGH, SOFT_LOW, SOFT_LOW},
    {SOFT_LOW, SOFT_LOW, SOFT_HIGH, SOFT_LOW},
    {SOFT_LOW, SOFT_LOW, SOFT_LOW, SOFT_HIGH}
  };

  MLP_Weights_Init();
  MLP_Train(4, 1000, 0.0001, pattern_set, pattern_set);
  MLP_Dump();

  return 0;
}
