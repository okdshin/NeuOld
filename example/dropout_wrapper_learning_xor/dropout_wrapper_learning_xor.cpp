#include <iostream>
#include <neu/layer_algorithm.hpp>
#include <neu/activate_func/rectifier.hpp>
#include <neu/activate_func/sigmoid.hpp>
#include <neu/activate_func/identity.hpp>
#include <neu/full_connected_layer.hpp>
#include <neu/dropout_wrapper.hpp>
#include <neu/kernel.hpp>
//#include <boost/timer/timer.hpp>

decltype(auto) print(neu::cpu_vector const& v) {
	for(auto const& e : v) {
		std::cout << e << " ";
	}
	std::cout << std::flush;
}
decltype(auto) print(neu::gpu_vector const& v) {
	print(neu::to_cpu_vector(v));
}
int main(int argc, char** argv) {
	//boost::timer::auto_cpu_timer t;
	std::cout << "hello world" << std::endl;

	std::random_device rd;
	std::mt19937 rand(rd());

	auto input_dim = 2u;
	auto output_dim = 1u;
	auto batch_size = 4u;
	std::vector<neu::cpu_vector> cpu_input = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};
	std::vector<neu::cpu_vector> cpu_teach = {
		{0.f}, {1.f}, {1.f}, {0.f}
	};

	neu::gpu_vector input;
	for(auto const& cpui : cpu_input) {
		input.insert(input.end(), cpui.begin(), cpui.end());
	}
	neu::gpu_vector teach;
	for(auto const& cput : cpu_teach) {
		teach.insert(teach.end(), cput.begin(), cput.end());
	}

	/*
	auto multiply_kernel = neu::make_kernel(neu::multiply_kernel_source, "multiply");
	auto multiply_back_kernel =
		neu::make_kernel(neu::multiply_back_kernel_source, "multiply_back");
	auto update_delta_weight_kernel =
		neu::make_kernel(neu::update_delta_weight_kernel_source, "update_delta_weight");
	*/

	std::uniform_real_distribution<> bin{-1.f, 1.f};

	auto fc0 = neu::make_full_connected_layer(input_dim, 10, batch_size, neu::rectifier());
	fc0.init_weight_randomly([&rand, &bin]() { return bin(rand); });

	auto fc1 = neu::make_full_connected_layer(10, 20, batch_size, neu::rectifier());
	fc1.init_weight_randomly([&rand, &bin]() { return bin(rand); });

	auto fc2 = neu::make_full_connected_layer(20, 10, batch_size, neu::rectifier());
	fc2.init_weight_randomly([&rand, &bin]() { return bin(rand); });

	auto fc3 = neu::make_full_connected_layer(10, output_dim, batch_size, neu::sigmoid());
	fc3.init_weight_randomly([&rand, &bin]() { return bin(rand); });

	auto layers = std::make_tuple(
		neu::make_dropout_wrapper(fc0, 0.9f, rand),
		neu::make_dropout_wrapper(fc1, 0.7f, rand),
		neu::make_dropout_wrapper(fc2, 0.5f, rand),
		fc3
	);
	std::cout << "weight "; print(std::get<0>(layers).get_layer().get_weight());
	std::cout << std::endl;

	for(auto i = 0u; i < 1000u; ++i) {
		neu::feedforward_x(layers, input);

		const auto y = neu::layer_get_y(neu::tuple_back(layers));
		neu::gpu_vector errors(y.size());
		boost::compute::transform(y.begin(), y.end(), teach.begin(), errors.begin(),
			boost::compute::minus<neu::scalar>());
		neu::feedback_delta(layers, errors);

		neu::update_delta_weight(layers, input);
		neu::update_weight(layers);

		neu::gpu_vector squared_errors(errors.size());
		boost::compute::transform(errors.begin(), errors.end(),
			errors.begin(), squared_errors.begin(),
			boost::compute::multiplies<neu::scalar>());
		auto error_sum = boost::compute::accumulate(
			squared_errors.begin(), squared_errors.end(), 0.f);
		if(i%100 == 0) {
			std::cout << i << ":" << error_sum << std::endl;
		}
	}
	print(teach); std::cout << "\n";
	neu::feedforward_x_without_train(layers, input);
	auto y = neu::layer_get_y(neu::tuple_back(layers));
	print(y);
	std::cout << std::endl;
}
