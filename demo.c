#include "stdio.h"
#include "mlp.h"

int main(){

  neurone_input[0] = 1;
  neurone_input[1] = 0;
  neurone_input[2] = 0;
  neurone_input[3] = 0;

  MLP_Weights_Init();
  MLP_Evaluate();
  MLP_Dump();

  return 0;
}
