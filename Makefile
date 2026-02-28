CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

symnmf: symnmf.o matrix_utils.o
	$(CC) -o symnmf symnmf.o matrix_utils.o $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h matrix_utils.h
	$(CC) $(CFLAGS) -c symnmf.c -o symnmf.o

matrix_utils.o: matrix_utils.c matrix_utils.h
	$(CC) $(CFLAGS) -c matrix_utils.c -o matrix_utils.o

clean:
	rm -f symnmf *.o
