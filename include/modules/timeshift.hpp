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
        uint64_t max_batch;
        uint64_t max_seq;
        uint64_t dims;
        
        TimeShift(){
        }

        TimeShift(const ulong max_batch, const ulong max_seq, const ulong dims){
            std::vector<uint64_t> size = {max_batch, max_seq, dims};
            this->buffer = Tensor<float>(size,0.0);
            std::vector<uint64_t> state_size = {max_batch, 1UL, dims};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor<float>(state_size,0.0);
            
            this->max_batch = max_batch;
            this->max_seq = max_seq;
            this->dims = dims;
            
        }

        Tensor<float> operator()(Tensor<float> input){
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