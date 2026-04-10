CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

# The default target that runs when you just type 'make'
all: symnmf

symnmf: symnmf.o matrix_utils.o
	$(CC) symnmf.o matrix_utils.o -o symnmf $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h matrix_utils.h
	$(CC) $(CFLAGS) -c symnmf.c

matrix_utils.o: matrix_utils.c matrix_utils.h
	$(CC) $(CFLAGS) -c matrix_utils.c

clean:
	# Added cleanup for python build artifacts to stay compliant with Section 3
	rm -f symnmf *.o *.so
	rm -rf build/