/* inspired by 
 * https://stackoverflow.com/questions/23324480
 */

#include <cblas.h>
#include <iostream> // HPC can into CPP!
#include <random>
#include <memory>
#include <cassert>
#include <chrono>

void matmult(int n, const double *A_raw, const double *B_raw, double *C_raw) {
    auto A_mem = std::unique_ptr<double[]>(new double[n*n]);
    auto B_mem = std::unique_ptr<double[]>(new double[n*n]);
    auto C_mem = std::unique_ptr<double[]>(new double[n*n]);
    double*__restrict__ A = A_mem.get();
    double*__restrict__ B = B_mem.get();
    double*__restrict__ C = C_mem.get();

    // Assumptions:
    // L3_step % SIMD == 0
    // L2_step % SIMD == 0
    // L1_step % SIMD == 0
    // n % L3_step == 0
    // n % L2_step == 0
    // n % L1_step == 0
    constexpr int SIMD = 4, L1_step = 32, L2_step = 32, L3_step = 32;

    //for (int y2 = 0; y2 < n; y2 += L3_step) {
        for (int s = 0; s < n; s += L1_step) {
            for (int x2 = 0; x2 < n; x2 += L2_step) {
                //for (int y1 = y2; y1 < y2 + L3_step; y1 += SIMD) {
                    for (int x1 = x2; x1 < x2 + L2_step; x1 += SIMD) {
                        auto AA = &A[s * n + x1 * L1_step];
                        for (int i = 0; i < L1_step; ++i) {
                            for (int m = 0; m < SIMD; ++m) {
                                //for (int k = 0; k < SIMD; ++k) {
                                    AA[SIMD*i + m] = A_raw[(x1 + m) * n + (s + i)];
                                //}
                            }
                        }
                    }
                //}
            }
        }
    //}

    for (int y2 = 0; y2 < n; y2 += L3_step) {
        for (int s = 0; s < n; s += L1_step) {
            //for (int x2 = 0; x2 < n; x2 += L2_step) {
                for (int y1 = y2; y1 < y2 + L3_step; y1 += SIMD) {
                    //for (int x1 = x2; x1 < x2 + L2_step; x1 += SIMD) {
                        auto BB = &B[y2 * n + s * L3_step + (y1 - y2) * L1_step];
                        for (int i = 0; i < L1_step; ++i) {
                            //for (int m = 0; m < SIMD; ++m) {
                                for (int k = 0; k < SIMD; ++k) {
                                    BB[SIMD*i + k] = B_raw[(s + i) * n + (y1 + k)];
                                }
                            //}
                        }
                    //}
                }
            //}
        }
    }

    for (int y2 = 0; y2 < n; y2 += L3_step) {
        for (int s = 0; s < n; s += L1_step) {
            for (int x2 = 0; x2 < n; x2 += L2_step) {
                for (int y1 = y2; y1 < y2 + L3_step; y1 += SIMD) {
                    for (int x1 = x2; x1 < x2 + L2_step; x1 += SIMD) {
                        auto AA = &A[s * n + x1 * L1_step];
                        auto BB = &B[y2 * n + s * L3_step + (y1 - y2) * L1_step];
                        auto CC = &C[y1 * n + x1 * SIMD];
                        double res[SIMD][SIMD] = {0};
                        for (int i = 0; i < L1_step; ++i) {
                            __builtin_prefetch(&AA[SIMD*i + SIMD * L1_step]);
                            //__builtin_prefetch(&BB[SIMD*i + SIMD * L1_step]);
                            for (int m = 0; m < SIMD; ++m) {
                                for (int k = 0; k < SIMD; ++k) {
                                    res[m][k] += AA[SIMD*i + m] * BB[SIMD*i + k];
                                }
                            }
                        }
                        for (int m = 0; m < SIMD; ++m) {
                            for (int k = 0; k < SIMD; ++k) {
                                CC[m*SIMD + k] += res[m][k];
                            }
                        }
                    }
                }
            }
        }
    }
    for (int y1 = 0; y1 < n; y1 += SIMD) {
        for (int x1 = 0; x1 < n; x1 += SIMD) {
            auto CC = &C[y1 * n + x1 * SIMD];
            for (int m = 0; m < SIMD; ++m) {
                for (int k = 0; k < SIMD; ++k) {
                    C_raw[(x1 + m) * n + y1 + k] = CC[m*SIMD + k];
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "invocation: " <<argv[0]<<" matrix_size " << std::endl;
        return 1;
    }
    const long long n{std::stoi(std::string{argv[1]})};
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist{0, 1};
    
    double* A = new double[n*n];
    double* B = new double[n*n];
    double* C = new double[n*n];
    double* C2 = new double[n*n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i*n + j] = doubleDist(rnd);
            B[i*n + j] = doubleDist(rnd);
            C[i*n + j] = 0;
            C2[i*n + j] = 0;
        }
    }

    auto startTime = std::chrono::steady_clock::now();
    auto finishTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed;
    for (int i = 0; i < 3; ++i) {
#if 1
        startTime = std::chrono::steady_clock::now();
        // this segfaults for matrix size 5000 and more
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n,
                    1.0, A, n, B, n, 0.0, C, n);
        finishTime = std::chrono::steady_clock::now();
        elapsed = finishTime - startTime;
        std::cout << "CBLAS elapsed time: "<< elapsed.count() << "[s]" << std::endl;
#endif
#if 1
        startTime = std::chrono::steady_clock::now();
        matmult(n, A, B, C2);
        finishTime = std::chrono::steady_clock::now();

        elapsed = finishTime - startTime;
        std::cout << "matmul elapsed time: "<< elapsed.count() << "[s]" << std::endl;
#endif
        for (int x = 0; x < n; ++x) {
            for (int y = 0; y < n; ++y) {
                assert(fabs(C[x*n + y] - C2[x*n + y]) < 0.000001);
            }
        }
    }
}

