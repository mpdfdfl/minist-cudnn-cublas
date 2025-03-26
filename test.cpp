#include <cuda_bf16.h>
#include <iostream>

int main()
{
    float f = 3.14f;

    // 使用 static_cast
    __nv_bfloat16 bf16_static = static_cast<__nv_bfloat16>(f);

    // 使用 __float2bfloat16
    __nv_bfloat16 bf16_cuda = __float2bfloat16(f);

    // 转回 float 检查数值
    float f_static = __bfloat162float(bf16_static);
    float f_cuda = __bfloat162float(bf16_cuda);

    std::cout << "Original float: " << f << std::endl;
    std::cout << "static_cast result: " << f_static << std::endl;
    std::cout << "__float2bfloat16 result: " << f_cuda << std::endl;

    return 0;
}
