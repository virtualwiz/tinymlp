CC = gcc
CFLAGS = -Wall
RM = rm -f

demo: mlp.o demo.o
	${CC} ${CFLAGS} mlp.o demo.o -lm -o demo

demo.o: demo.c
	${CC} ${CFLAGS} -c demo.c

mlp.o: mlp.c mlp.h
	${CC} ${CFLAGS} -c mlp.c

clean:
	${RM} *.o
	${RM} demo
