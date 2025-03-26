#include "mnist.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char *argv[])
{
    /* configure the network */
    int batch_size_train = 256; // 批次大小
    int num_steps_train = 2400;
    int monitoring_step = 200;

    double initial_learning_rate = 0.02f;
    double learning_rate = 0.0;
    double lr_decay = 0.0005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST<float> train_data_loader_float = MNIST<float>("./dataset");
    train_data_loader_float.train(batch_size_train, true);
    MNIST<half> train_data_loader_fp16 = MNIST<half>("./dataset");
    train_data_loader_fp16.train(batch_size_train, true);
    MNIST<__nv_bfloat16> train_data_loader_bf16 = MNIST<__nv_bfloat16>("./dataset");
    train_data_loader_bf16.train(batch_size_train, true);
    return 0;
}
