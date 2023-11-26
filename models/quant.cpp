#include <torch/extension.h>
#include "ATen/ATen.h"
#include <immintrin.h>
#include <algorithm>
#include <omp.h>

void matmul_avx512_optimized(const torch::Tensor &At, const torch::Tensor &Art, const torch::Tensor &Aot,
                             const torch::Tensor &Bt, torch::Tensor &Ct,
                             const long BB, const long IN, const long T, const long OUT)
{
    // Pointers to the data
    auto A = At.data_ptr<unsigned char>();
    auto Ar = Art.data_ptr<float>();
    auto Ao = Aot.data_ptr<float>();
    auto B = Bt.data_ptr<float>();
    auto C = Ct.data_ptr<float>();

// Parallel computation
#pragma omp parallel for collapse(2) schedule(guided,64) shared(A, Ar, Ao, B, C)
    
    for (long bbj = 0; bbj < BB * T; bbj += 1)
    {
    for (long i = 0; i < OUT; i += 16)
        {
    
            // __m128 testacc = _mm128_setzero_ps();
            __m512 acc = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();
            __m512 acc8 = _mm512_setzero_ps();
            __m512 acc9 = _mm512_setzero_ps();
            __m512 acc10 = _mm512_setzero_ps();
            __m512 acc11 = _mm512_setzero_ps();
            __m512 acc12 = _mm512_setzero_ps();
            __m512 acc13 = _mm512_setzero_ps();
            __m512 acc14 = _mm512_setzero_ps();
            __m512 acc15 = _mm512_setzero_ps();
            __m512 acc16 = _mm512_setzero_ps();
            float* scale = &Ar[i];
            float* offset = &Ao[i];

            #pragma unroll(16)
            for (long k = 0; k < IN; k += 16)
            {  
                __m512 cxx = _mm512_load_ps(B + bbj * IN + k);
                u_int8_t* aink = A + i * IN + k;
               
                acc = _mm512_fmadd_ps(offset[0]+scale[0]* _mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(aink))),cxx,acc);                 
                acc2 = _mm512_fmadd_ps(offset[1]+scale[1]* _mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(aink + IN))),cxx,acc2); 
                acc3 = _mm512_fmadd_ps(offset[2]+scale[2]* _mm512_cvtepi32_ps(
                                        _mm512_cvtepu8_epi32(*(__m128i *)(aink + IN*2))),cxx,acc3); 
                acc4 = _mm512_fmadd_ps(offset[3]+scale[3]* _mm512_cvtepi32_ps(
                                        _mm512_cvtepu8_epi32(*(__m128i *)(aink + IN*3))),cxx,acc4); 
                acc5 = _mm512_fmadd_ps(offset[4]+scale[4]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+4) * IN + k))),
                           cxx,
                            acc5);
                acc6 = _mm512_fmadd_ps(offset[5]+scale[5]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+5) * IN + k))),
                           cxx,
                            acc6);
                acc7 = _mm512_fmadd_ps(offset[6]+scale[6]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+6) * IN + k))),
                           cxx,
                            acc7);
                acc8 = _mm512_fmadd_ps(offset[7]+scale[7]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+7) * IN + k))),
                           cxx,
                            acc8);
                acc9 = _mm512_fmadd_ps(offset[8]+scale[8]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+8) * IN + k))),
                           cxx,
                            acc9);
                acc10 = _mm512_fmadd_ps(offset[9]+scale[9]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+9) * IN + k))),
                           cxx,
                            acc10);
                acc11 = _mm512_fmadd_ps(offset[10]+scale[10]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+10) * IN + k ))),
                           cxx,
                            acc11);
                acc12 = _mm512_fmadd_ps(offset[11]+scale[11]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+11) * IN + k ))),
                           cxx,
                            acc12);
                acc13 = _mm512_fmadd_ps(offset[12]+scale[12]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+12) * IN + k ))),
                           cxx,
                            acc13);
                acc14 = _mm512_fmadd_ps(offset[13]+scale[13]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+13) * IN + k ))),
                           cxx,
                            acc14);
                acc15 = _mm512_fmadd_ps(offset[14]+scale[14]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+14) * IN + k ))),
                           cxx,
                            acc15);
                acc16 = _mm512_fmadd_ps(offset[15]+scale[15]*_mm512_cvtepi32_ps(
                                            _mm512_cvtepu8_epi32(*(__m128i *)(A + (i+15) * IN + k ))),
                           cxx,
                            acc16);



            }
            __m512 out = _mm512_set_ps(
                _mm512_reduce_add_ps(acc16),
                _mm512_reduce_add_ps(acc15),
                _mm512_reduce_add_ps(acc14),
                _mm512_reduce_add_ps(acc13),
                _mm512_reduce_add_ps(acc12),
                _mm512_reduce_add_ps(acc11),
                _mm512_reduce_add_ps(acc10),
                _mm512_reduce_add_ps(acc9),
                _mm512_reduce_add_ps(acc8),
                _mm512_reduce_add_ps(acc7),
                _mm512_reduce_add_ps(acc6),
                _mm512_reduce_add_ps(acc5),
                _mm512_reduce_add_ps(acc4),
                _mm512_reduce_add_ps(acc3),
                _mm512_reduce_add_ps(acc2),
                _mm512_reduce_add_ps(acc));

            _mm512_store_ps(C + bbj * OUT + i,out);
            
            
            
        }
    }
}
void matmul_avx512_optimized_float(const torch::Tensor &At,
                                   const torch::Tensor &Bt, torch::Tensor &Ct,
                                   const long BB, const long IN, const long T, const long OUT)
{
    // Pointers to the data
    auto A = At.data_ptr<float>();
    auto B = Bt.data_ptr<float>();
    auto C = Ct.data_ptr<float>();

// Parallel computation
#pragma omp parallel for collapse(2) schedule(dynamic, 256) shared(A, B, C)
    for (long i = 0; i < OUT; i += 1)
    {

        for (long bbj = 0; bbj < BB * T; bbj += 1)
        {

            __m512 acc = _mm512_setzero_ps();
#pragma unroll(16)
            for (long k = 0; k < IN; k += 16)
            {

                acc = _mm512_fmadd_ps(
                    *(__m512 *)(A + i * IN + k),
                    *(__m512 *)(B + bbj * IN + k),
                    acc);
        
            }
            *(C + bbj * OUT + i) += _mm512_reduce_add_ps(acc);
        }
    }
}

void Quantize(torch::Tensor &At, torch::Tensor &Art, torch::Tensor &Aot, torch::Tensor &Aqt, long M, long N)
{
    float *A = At.data_ptr<float>();
    float *Ar = Art.data_ptr<float>();
    float *Ao = Aot.data_ptr<float>();
    u_char *Aq = Aqt.data_ptr<u_char>();

    long i, j;
    for (i = 0; i < M; i++)
    {
        __m512 amax = _mm512_set1_ps(-1e9);
        __m512 amin = _mm512_set1_ps(1e9);
        for (j = 0; j < N; j += 16)
        {
            __m512 a = _mm512_load_ps(A + i * N + j);
            amax = _mm512_max_ps(amax, a);
            amin = _mm512_min_ps(amin, a);
        }
        float max = _mm512_reduce_max_ps(amax);
        float min = _mm512_reduce_min_ps(amin);
        float range = (max - min);
        float scale = (range/255);
        *(Ar + i)= scale;
        *(Ao + i)= min;
        for (j = 0; j < N; j += 16)
        {
            __m512 a = _mm512_load_ps(A + i * N + j);

            __m512 d = _mm512_div_ps(_mm512_sub_ps(a, _mm512_set1_ps(min)), _mm512_set1_ps(scale));

            for (long k = 0; k < 16; k++)
            {
                // std::cout << d[k] << ":" << long(d[k]) << ":" << int((u_char)(int(d[k]))) << ":" << int((u_char)((unsigned int)(d[k]))) << std::endl;
                Aq[i * N + j + k] = (u_int8_t)((u_int32_t)(d[k]));
            }
        }
    }
}

// pytorch bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_cpu", &Quantize, "QuantizeCpu");
    m.def("matmul", &matmul_avx512_optimized, "matmul_avx512I");
    m.def("matmul_float", &matmul_avx512_optimized_float, "matmul_avx512F");
}

TORCH_LIBRARY(wkv5, m)
{
    m.def("quantize_cpu", Quantize);
    m.def("matmul", matmul_avx512_optimized);
    m.def("matmul_float", matmul_avx512_optimized_float);
}