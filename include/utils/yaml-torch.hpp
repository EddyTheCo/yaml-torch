#include<yaml-cpp/yaml.h>
#include <ATen/ATen.h>
#include <torch/torch.h>


namespace custom_models{
namespace yaml_interface{


std::unique_ptr<torch::optim::Optimizer> get_optimizer(YAML::Node config,const  std::vector<at::Tensor> &params);

}

}
