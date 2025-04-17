#include "pch.h"

#include "Tensor.h"
#include "TensorBackend.h"

INITIALIZE_EASYLOGGINGPP

int main() {
    el::Configurations conf("E:/.vscode/SaveCode/c++/NForge/NForge/src/log.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    LOG(INFO) << "NForge has started!";

    Tensor a({ 10 }, 1.0f, Backend::CUDA);
    Tensor b({ 10 }, 3.0f, Backend::CUDA);

    std::cout << "a shape: " << a.getShapeAsString() << " content: " << a.getDataAsString() << std::endl;
    std::cout << "b shape: " << b.getShapeAsString() << " content: " << b.getDataAsString() << std::endl;

    (a + b).print();
    (a - b).print();
    (a * b).print();
    (a / b).print();

    return 0;
}