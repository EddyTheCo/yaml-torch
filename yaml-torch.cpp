#include"utils/yaml-interface.hpp"

namespace custom_models{
namespace yaml_interface{

std::unique_ptr<torch::optim::Optimizer> get_optimizer(YAML::Node config,const std::vector<torch::Tensor>& params)
{
    if(config["Adam"])
    {
        return std::unique_ptr<torch::optim::Optimizer>(
                    new torch::optim::Adam(params,torch::optim::AdamOptions()
                                           .lr((config["Adam"])["lr"].as<double>())));
    }
    if(config["SGD"])
    {
        return std::unique_ptr<torch::optim::Optimizer>(
                    new torch::optim::SGD(params,torch::optim::SGDOptions((config["SGD"])["lr"].as<double>())
                                           .momentum((config["SGD"])["momentum"].as<double>())));
    }
    return std::unique_ptr<torch::optim::Optimizer>(
                new torch::optim::Adam(params,torch::optim::AdamOptions()
                                       .lr(0.1)));
}
}
}
