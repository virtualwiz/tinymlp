demo: mlp.o demo.o
	gcc -Wall -lm mlp.o demo.o -o demo

demo.o: demo.c
	gcc -Wall -c demo.c

mlp.o: mlp.c mlp.h
	gcc -Wall -c mlp.c

clean:
	rm *.o
	rm demo
