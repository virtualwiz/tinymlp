#include "stdio.h"
#include "mlp.h"

int main(){
  double x,y;
  scanf("%lf", &x);
  y = sigmoid(x);
  printf("%lf\n", y);
  return 0;
}
