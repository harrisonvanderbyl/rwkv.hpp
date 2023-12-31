#include <torch/extension.h>
#include "ATen/ATen.h"
#include <algorithm>
#include <omp.h>


void Quantize(torch::Tensor &At, torch::Tensor &Art, torch::Tensor &Aot, torch::Tensor &Aqt, long M, long N, bool addResiduals = true)
{
    float *A = At.data_ptr<float>();
    float *Ar = Art.data_ptr<float>();
    float *Ao = Aot.data_ptr<float>();
    u_char *Aq = Aqt.data_ptr<u_char>();

    long i, j;
    for (i = 0; i < M; i++)
    {
        float amax = (-1e9);
        float amin = (1e9);
        for (j = 0; j < N; j += 1)
        {
            float a = *(A + i * N + j);
            amax = std::max(amax, a);
            amin = std::min(amin, a);
        }
        float max = (amax);
        float min = (amin);
        float range = (max - min);
        uint BITS = 255;
        float scale = (range/BITS);
        *(Ar + i)= scale;
        *(Ao + i)= min;

        float diff = 0.0;
        for (j = 0; j < N; j += 1)
        {
            float a = *(A + i * N + j);

            float d = ((a - (min))/(scale));

            diff += (d - float((int)(d)));
            
                // std::cout << d[k] << ":" << long(d[k]) << ":" << int((u_char)(int(d[k]))) << ":" << int((u_char)((unsigned int)(d[k]))) << std::endl;
            Aq[i * N + j] = (u_int8_t)((u_int32_t)(d));
            
        }

        if (addResiduals){
            diff = diff / N;
            auto offset = (diff * scale);
            *(Ao + i) = *(Ao + i) + offset;
            
        }
    }
}

// pytorch bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_cpu", &Quantize, "QuantizeCpu");
}

TORCH_LIBRARY(wkv5, m)
{
    m.def("quantize_cpu", Quantize);
}