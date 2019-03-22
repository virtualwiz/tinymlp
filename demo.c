/* Copyright (C) 2019 S Du, R Jiao, University of Birmingham.
   This program is part of tinyMLP project: simple multi layer
   perceptron library and demonstration program.
   This program implemented, trained and tested an autoencoder
   neural network with an example dataset.
   The patterns are to be automatically binary-encoded.
*/

#include "mlp.h"
#include "stdio.h"
#include "stdlib.h"

/* Set to 1 to write errors into a .csv file,
   for plotting and observation. */
#define DATA_LOG_MODE 0
#define REPORT_INTERVAL 10000

/* Set to 1 to enable auto reset feature,
   retry from a different initialisation of weight matrices
   after NUM_EPOCHS epochs. */
#define AUTO_RESET_MODE 1
#define NUM_EPOCHS 3000000
#define TARGET_ERROR 0.005

/* Training and testing sets size(number of patterns) */
#define NUM_PATTERNS 4

/* Learning rate(between 0 and 1) */
#define LEARNING_RATE 0.01

int main(){
  /* Training and testing set.
     In an autoencoder they are same.
     TODO: Load pattern sets from external files. */
  double pattern_set[NUM_PATTERNS * NUM_NEURONES_INPUT] = {
    SOFT_HIGH, SOFT_LOW, SOFT_LOW, SOFT_LOW,
    SOFT_LOW, SOFT_HIGH, SOFT_LOW, SOFT_LOW,
    SOFT_LOW, SOFT_LOW, SOFT_HIGH, SOFT_LOW,
    SOFT_LOW, SOFT_LOW, SOFT_LOW, SOFT_HIGH,
  };

  int i, j;
  unsigned long int i_epoch = 0;
  double mlp_err;

#if DATA_LOG_MODE
  /* Open error log file. */
  FILE* logfile_ptr;
  logfile_ptr = fopen("error_log.csv", "w");
  if(logfile_ptr == NULL){
    printf("I/O Error: Cannot open log file.");
  }
#endif

  /* Initialise the neural network. */
  MLP_Init();

  mlp_err = MLP_ErrorAvg(NUM_PATTERNS, pattern_set, pattern_set);
  printf("Initial avg error is %lf\n", mlp_err);
#if DATA_LOG_MODE
  fprintf(logfile_ptr, "0,%lf\n", mlp_err);
#endif

  /* Start training */
  for(;;){
    MLP_Train(NUM_PATTERNS, LEARNING_RATE, pattern_set, pattern_set);
    i_epoch += 1;
    mlp_err = MLP_ErrorAvg(NUM_PATTERNS, pattern_set, pattern_set);
    if(!(i_epoch % REPORT_INTERVAL)){
      printf("Avg error is %lf\tafter %ld epochs\n", mlp_err, i_epoch);
#if DATA_LOG_MODE
      fprintf(logfile_ptr, "%ld,%lf\n", i_epoch, mlp_err);
#endif
    }
#if AUTO_RESET_MODE
    if(mlp_err <= TARGET_ERROR){
      break;
    }
    else if(i_epoch == NUM_EPOCHS){
      /* Retry from another point in error space */
      printf("Hit epochs limit, resetting weights...\n");
      MLP_Init();
      i_epoch = 0;
    }
#else
    if(mlp_err <= TARGET_ERROR){
      break;
    }
#endif
  }

  /* Test the neural network with patterns */
  MLP_Dump(0);
  for(j = 0; j < NUM_PATTERNS; j++){
    for(i = 0; i < NUM_NEURONES_INPUT; i++){
      neurone_input[i] = pattern_set[NUM_NEURONES_INPUT * j + i];
    }
    MLP_Evaluate();
    MLP_Dump(1);
  }

#if DATA_LOG_MODE
  /* Close log file */
  fclose(logfile_ptr);
#endif

  return 0;
}
