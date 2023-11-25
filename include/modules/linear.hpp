#ifndef LINEAR_HPP
#define LINEAR_HPP
#include "hvml/tensor.hpp"
#include "safetensors/safetensors.hpp"
#include <iostream>
class Linear
{
    public:
        Tensor<bfloat16> weight;
        Tensor<float> range;
        Tensor<float> offset;
        Tensor<float> buffer;
        bool quantized = false;
        bool isbf16 = false;

        
        Linear(){
            
        }

        Linear(safetensors::safetensors_t& model, std::string prefix, ulong max_batch, ulong max_seq){
            try {
                this->weight = model.getBF16(prefix + ".weight");
                this->isbf16 = true;
            }
            catch (std::exception& e){
                // std::cout << "Exception:" << e.what() << std::endl;
                // std::cout << "Linear:" << prefix << " is not BF16" << std::endl;
                auto temp = model[prefix + ".weight"];
                this->weight = *(Tensor<bfloat16>*)&temp;
                this->isbf16 = false;
                
            }
            if (model.contains(prefix + ".range")){
                this->range = model[prefix + ".range"];
                this->offset = model[prefix + ".offset"];
                this->quantized = true;
            }

            this->buffer = Tensor<float>({max_batch, max_seq, this->weight.shape[0]});
        }

        // Copy constructor
        Linear(const Linear& other){
            this->weight = other.weight;
            this->range = other.range;
            this->offset = other.offset;
            this->quantized = other.quantized;
            this->buffer = other.buffer;
            this->isbf16 = other.isbf16;
            
        }
        
        Tensor<float> operator()(Tensor<float>& input) {
               if (this->quantized){
                    throw std::runtime_error("Quantized Linear not implemented");
                }
                else{
                    auto mbuff = Tensor<float>({input.shape[0], input.shape[1], this->weight.shape[0]},
                        this->buffer.data);
                    if(this->isbf16){
                        ((Tensor<bfloat16>*)(&this->weight))->matmul(input, mbuff, true);
                    }
                    else{
                        ((Tensor<float>*)(&this->weight))->matmul(input, mbuff);
                    }
                    return mbuff;
                }     
        }

};

#endif