#include <iostream>
#include <string>
#include <fstream>
#include "safetensors/safetensors.hpp"
#include "modules/embedding.hpp"
#include "modules/layernorm.hpp"
#include "modules/linear.hpp"
#include "modules/block.hpp"
class RWKV
{
    Embedding emb1;
    LayerNorm ln0;
    LayerNorm ln_out;
    Linear output;
    std::vector<Block> blocks;

public:
    ulong layers;

    RWKV(std::string path, ulong max_batch = 1, ulong max_seq = 50)
    {
        std::ifstream inFile;
        inFile.open(path, std::ios::binary);
        auto model = safetensors::deserialize(inFile);


        auto keys = model.keys();
        layers = 0;
        for (auto key : keys)
        {
            if (std::string(key).find("blocks.") != std::string::npos)
            {
                if (std::string(key).find("att.key") != std::string::npos)
                {
                    layers++;
                }
               
            }
        }

        // std::cout << "layers:" << layers << std::endl;

        auto t1 = model["emb.weight"];
        this->emb1 = Embedding(t1, max_batch, max_seq);
        this->ln0 = LayerNorm(model["blocks.0.ln0.weight"], model["blocks.0.ln0.bias"], max_batch, max_seq);
        this->ln_out = LayerNorm(model["ln_out.weight"], model["ln_out.bias"], max_batch, max_seq);
        this->output = Linear(model, "head", max_batch, max_seq);
        for (int i = 0; i < layers; i++)
        {
            blocks.push_back(Block(model, i, max_batch, max_seq));
        }
    }

    Tensor<float> operator()(std::vector<std::vector<ulong>> input)
    {
        auto x = emb1(input);
        x = ln0(x);
        for (int i = 0; i < layers; i++)
        {
            x = blocks[i](x);
        }
        x = ln_out(x);
        auto t3 = output(x);
        return t3;
    }
};
