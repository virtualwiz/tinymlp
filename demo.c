#include "stdio.h"
#include "mlp.h"

int main(){
  int i;

  neurone_input[0] = HIGH;
  neurone_input[1] = LOW;
  neurone_input[2] = LOW;
  neurone_input[3] = LOW;

  MLP_Weights_Init();
  MLP_Evaluate();

  for(i = 0; i <= NUM_NEURONES_OUTPUT; i++){
    printf("%lf\t", neurone_output[i]);
  }

  return 0;
}
