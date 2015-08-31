#include <iostream>
#include <neu/vector_io.hpp>
#include <neu/layer_algorithm.hpp>
#include <neu/kernel.hpp>
#include <neu/kernel.hpp>
#include <neu/activate_func/sigmoid.hpp>
#include <neu/activate_func/rectifier.hpp>
#include <neu/full_connected_layer.hpp>

int main(int argc, char** argv) {
	std::cout << "hello world" << std::endl;

	std::random_device rand{};

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

	auto layers = std::make_tuple(
		neu::make_full_connected_layer(input_dim, 3, batch_size, neu::rectifier()),
		neu::make_full_connected_layer(3, output_dim, batch_size, neu::sigmoid())
	);

	std::uniform_real_distribution<> bin{-1,1};
	neu::tuple_foreach(layers, [&rand, &bin](auto& l){
		l.init_weight_randomly([&rand, &bin]() { return bin(rand); }); });
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
			squared_errors.begin(), squared_errors.end(), 0.0f);
		if(i%100 == 0) {
			std::cout << i << ":" << error_sum << std::endl;
		}
	}
	neu::print(teach); std::cout << "\n";
	auto y = std::get<std::tuple_size<decltype(layers)>::value-1>(layers).get_y();
	neu::print(y);
	std::cout << std::endl;
}
