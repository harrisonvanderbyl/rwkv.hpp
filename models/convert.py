import safetensors.torch as st
import torch


# change path to your model
path = "./3B.pth"
model = torch.load(path, "cpu")

import inquirer

questions = [
    inquirer.List('mode',
                message="What do you want to do?",
                choices=['Convert to BF16', 'Convert to FP32'],
            ),
]

bf16 = inquirer.prompt(questions)["mode"] == "Convert to BF16"

import cpuinfo
hasbf16 = "avx512_bf16" in cpuinfo.get_cpu_info_json()
avx512 = "avx512" in cpuinfo.get_cpu_info_json()
avx2 = "avx2" in cpuinfo.get_cpu_info_json()
neon = "neon" in cpuinfo.get_cpu_info_json()


import tqdm as tqdm
for key in tqdm.tqdm(model.keys()):
    if model[key].shape.__len__() == 2 and key != "emb.weight" and "time_" not in key:
        
        # bf16 conversion for avx512
        if bf16:
            model[key] = model[key].bfloat16().clone().cpu()
            shape = model[key].shape
            model[key] = model[key].reshape(-1,2,16)[:,[1,0]].reshape(shape)
        else:
            model[key] = model[key].float().clone().cpu()
    elif model[key].shape.__len__() == 1:
        model[key] = model[key].float().cpu()
    else:
        model[key] = model[key].float().cpu()
    if "decay" in key:
        model[key] = model[key].double().exp().neg().exp().float().cpu()
# create ../build/ if not exists
import os
if not os.path.exists("../build/"):
    os.makedirs("../build/")
st.save_file(model, "../build/model.safetensors")