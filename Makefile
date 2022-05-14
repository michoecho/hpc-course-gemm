CC := clang++
LFLAGS :=
#ALL := blas-dmmmult laplace-seq
ALL := blas-dmmmult

all : $(ALL)

blas-dmmmult: blas-dmmmult.cpp
	$(CC) -ffp-model=fast -g -march=native -O3 -v -o $@ $< -lcblas
#
#laplace-seq: laplace-seq.o laplace-common.o
#	$(CC) $(LFLAGS) -o $@ $^
#
#laplace-common.o: laplace-common.cpp Makefile
#	$(CC) -c $<
#
#laplace-seq.o: laplace-seq.cpp Makefile
#	$(CC) -c $<

clean :
	rm -f $(ALL)
