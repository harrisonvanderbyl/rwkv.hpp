#ifndef TIMESHIFT_HPP
#define TIMESHIFT_HPP
#include "hvml/tensor.hpp"
#include "safetensors/safetensors.hpp"
class TimeShift
{
    public:
        uint shiftamount = 1;
        Tensor<float> state;
        Tensor<float> buffer;
        
        TimeShift(){
        }

        TimeShift(ulong max_batch, ulong max_seq, ulong dims){
            this->buffer = Tensor<float>({max_batch, max_seq, dims});
            this->state = Tensor<float>({max_batch, 1, dims},0.0f);
        }

        Tensor<float> operator()(Tensor<float>& input){
            auto out = Tensor<float>({input.shape[0], input.shape[1], input.shape[2]}, this->buffer.data);
            auto batches = input.shape[0];
            auto seq = input.shape[1];
            for (int i = 0; i < batches; i++){
                out[i][0].clone(this->state[i][0]);
                for (int j = 0; j < seq; j++){
                    if (j > 0){
                        out[i][j].clone(input[i][j-1]);
                    }
                    else{
                        this->state[i][0].clone(input[i][seq-1]);
                    }
                }
            }
            return out;            
        }

};

#endif