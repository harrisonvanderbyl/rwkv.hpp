// simd

#ifndef INTRINSICS_HPP
#define INTRINSICS_HPP
// get pow
#include <cmath>

#ifdef __AVX512F__  // This macro is defined if AVX-512 is supported
  #include <immintrin.h>
  
  #define SIMD_WIDTH 16
  #define LOAD(x) _mm512_loadu_ps(x)
  #define STORE(x, y) _mm512_storeu_ps(x, y)
  #define SET1(x) _mm512_set1_ps(x)
  #define MULTIPLY(x, y) _mm512_mul_ps(x, y)
  #define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
  #define REDUCE(x) _mm512_reduce_add_ps(x)
  #define ADD(x, y) _mm512_add_ps(x, y)
  #define MAX(x, y) _mm512_max_ps(x, y)
  #define SIMDTYPE __m512
  #define EXP(x) exp_ps_fill(x)
  SIMDTYPE exp_ps_fill(SIMDTYPE x){
    SIMDTYPE result = SET1(0.0f);
    for (int i = 0; i < SIMD_WIDTH; i++){
        result[i] = pow(M_E, x[i]);
    }
    return result;
}
  
  
  #define DIVIDE(x, y) _mm512_div_ps(x, y)

  // check if bf16 is supported
    #ifdef __AVX512BF16__
        #pragma message("AVX-512-bf16 is supported")
        #define LOADBF16(x) (__m512bh)_mm512_loadu_si512(x)
        // load 2 continous fp32 values from memory and convert to bf16
        #define LOADFP32BF16(x) (__m512bh)_mm512_cvtne2ps_pbh(LOAD(x), LOAD(x + 16))
        // do dot product of 2 bf16 vectors
        #define DOTBF16(x, y, acc) _mm512_dpbf16_ps(acc, x, y)
        #define DOTBF16F32(x, y, acc) _mm512_dpbf16_ps(acc, x, y)

    #else
        #pragma message("AVX-512-bf16 is not supported, doing in place conversion")
        #define LOADBF16(x) x
        #define LOADFP32BF16(x) x

        // convert bf16 to fp32 by going uint16 -> int32(uint16, zeros) -> cast to float
        #define bf16_to_fp32(x) (__m512)_mm512_slli_epi32(_mm512_cvtepi16_epi32(*(__m256i*)(x)), 16)

        #define DOTBF16(x, y, acc) (_mm512_fmadd_ps(bf16_to_fp32(x+16), LOAD(y), _mm512_fmadd_ps(bf16_to_fp32(x), LOAD(y+16), acc)))

        #define DOTBF16F32(x, y, acc) (_mm512_fmadd_ps(LOAD(x), LOAD(y), _mm512_fmadd_ps(LOAD(x+16), LOAD(y+16), acc)))
    
    #endif
    // print out the SIMD width
  #pragma message("AVX-512 is supported")
#else
  // Fallback to AVX2 if AVX-512 is not supported
  #ifdef __AVX2__
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define LOAD(x) _mm256_load_ps(x)
    #define STORE(x, y) _mm256_store_ps(x, y)
    #define SET1(x) _mm256_set1_ps(x)
    #define MULTIPLY(x, y) _mm256_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
    #ifdef _mm256_reduce_add_ps
        #define REDUCE(x) _mm256_reduce_add_ps(x)
    #else
        #define REDUCE(x) x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]
    #endif
    #define ADD(x, y) _mm256_add_ps(x, y)
    #define MAX(x, y) _mm256_max_ps(x, y)
    #define DIVIDE(x, y) _mm256_div_ps(x, y)
    #define SIMDTYPE __m256
    #define EXP(x) exp_ps_fill(x)
    SIMDTYPE exp_ps_fill(SIMDTYPE x){
    SIMDTYPE result = SET1(0.0f);
    for (int i = 0; i < SIMD_WIDTH; i++){
        result[i] = pow(M_E, x[i]);
    }
    return result;
}
    // print out the SIMD width
    #pragma message("AVX-2 is supported")

    #define LOADBF16(x) x
    #define LOADFP32BF16(x) x

    // convert bf16 to fp32 by going uint16 -> int32(uint16, zeros) -> cast to float
    #define bf16_to_fp32(x) (__m256)_mm256_slli_epi32(_mm256_cvtepi16_epi32(*(__m128i*)(x)), 16)

    #define DOTBF16(x, y, acc) (_mm256_fmadd_ps(bf16_to_fp32(x+16), LOAD(y),_mm256_fmadd_ps(bf16_to_fp32(x+24), LOAD(y+8), _mm256_fmadd_ps(bf16_to_fp32(x), LOAD(y+16), _mm256_fmadd_ps(bf16_to_fp32(x+8), LOAD(y+24),acc)))))

    #define DOTBF16F32(x, y, acc) (_mm256_fmadd_ps(LOAD(x), LOAD(y), _mm256_fmadd_ps(LOAD(x+8), LOAD(y+8), _mm256_fmadd_ps(LOAD(x+16), LOAD(y+16),_mm256_fmadd_ps(LOAD(x+24), LOAD(y+24),acc)))))
    

  #else
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
        #define SIMD_WIDTH 4  // NEON typically operates on 128-bit registers (4 floats)
        #define LOAD(x) vld1q_f32(x)
        #define STORE(x, y) vst1q_f32(x, y)
        #define SET1(x) vdupq_n_f32(x)
        #define MULTIPLY(x, y) vmulq_f32(x, y)
        #define MULTADD(x, y, z) vmlaq_f32(z, x, y)
        #define REDUCE(x) vaddq_f32(x)
        #define ADD(x, y) vaddq_f32(x, y)
        #define MAX(x, y) vmaxq_f32(x, y)
        #define DIVIDE(x, y) vdivq_f32(x, y)
        #define SIMDTYPE float32x4_t
        #define EXP(x) exp_ps_fill(x)
        SIMDTYPE exp_ps_fill(SIMDTYPE x){
            SIMDTYPE result = SET1(0.0f);
            for (int i = 0; i < SIMD_WIDTH; i++){
                result[i] = pow(M_E, x[i]);
            }
            return result;
        }

        
        // Print out the SIMD width
        #pragma message("ARM NEON is supported")
    #else
        #pragma message("No SIMD is supported")
        #define SIMD_WIDTH 1
        #define LOAD(x) *(x)
        #define STORE(x, y) *(x) = y
        #define SET1(x) x
        #define MULTIPLY(x, y) (x * y)
        #define MULTADD(x, y, z) (x * y + z)
        #define ADD(x, y) (x + y)
        #define REDUCE(x) x
        #define MAX(x, y) (x > y ? x : y)
        #define EXP(x) exp(x)
        #define DIVIDE(x, y) (x / y)
        #define SIMDTYPE float
        #define EXP(x) exp(x)
    #endif

    #endif
#endif

#ifdef DEBUG
    #define DEBUG_MESSAGE(x) std::cout << x << std::endl;
#else
    #define DEBUG_MESSAGE(x)
#endif




#endif