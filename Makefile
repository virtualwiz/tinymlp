CC := gcc
CFLAGS := -Wall
RM := rm -f
LDFLAGS := -lm

demo: mlp.o demo.o
	${CC} ${CFLAGS} $^ $(LDFLAGS) -o $@

demo.o: demo.c
	${CC} ${CFLAGS} -c $^

mlp.o: mlp.c mlp.h
	${CC} ${CFLAGS} -c $<

clean:
	${RM} *.o
	${RM} demo
