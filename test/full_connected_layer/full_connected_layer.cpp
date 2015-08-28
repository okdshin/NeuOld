//TODO

#include <neu/kernel.hpp>
#include <neu/activate_func/identity.hpp>
#include <neu/full_connected_layer.hpp>

namespace neu_test {
	decltype(auto) full_connected_layer_test() {
	}
}

#ifndef NEU_TEST_CPP_INCLUDE
#	define NEU_TEST_TEST_TARGET_FUNCTION neu_test::full_connected_layer_test
#	include <test/include_main.hpp>
#endif

/*
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
	std::cout << "hello world" << std::endl;

	std::random_device rand{};

	auto input_dim = 2u;
	auto output_dim = 1u;
	auto batch_size = 4u;
	std::vector<neu::cpu_vector> cpu_input = {
		{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}
	};
	std::vector<neu::cpu_vector> cpu_teach = {
		{0.f}, {0.f}, {0.f}, {1.f}
	};

	neu::gpu_vector input;
	for(auto const& cpui : cpu_input) {
		input.insert(input.end(), cpui.begin(), cpui.end());
	}
	neu::gpu_vector teach;
	for(auto const& cput : cpu_teach) {
		teach.insert(teach.end(), cput.begin(), cput.end());
	}

	auto multiply_kernel = neu::make_kernel(neu::multiply_kernel_source, "multiply");
	auto multiply_back_kernel =
		neu::make_kernel(neu::multiply_back_kernel_source, "multiply_back");
	auto update_delta_weight_kernel =
		neu::make_kernel(neu::update_delta_weight_kernel_source, "update_delta_weight");
	auto full_connected = neu::make_full_connected_layer(
		input_dim, output_dim, batch_size, neu::sigmoid(),
		multiply_kernel, multiply_back_kernel, update_delta_weight_kernel);
	std::uniform_real_distribution<> bin{-1,1};
	full_connected.init_weight_randomly([&rand, &bin]() { return bin(rand); });
	auto weight = full_connected.get_weight();
	auto bias = full_connected.get_bias();
	std::cout << "weight"; print(weight); std::cout << "\n";
	std::cout << "bias"; print(bias); std::cout << "\n";
	for(auto i = 0u; i < 1000u; ++i) {
		std::cout << "epoch" << i;
		full_connected.calc_u_and_y(input);
		auto u = full_connected.get_u();
		auto y = full_connected.get_y();
		auto bias = full_connected.get_bias();
		std::cout << "u:"; print(u);
		std::cout << "y:"; print(y);
		std::cout << "bias:"; print(bias);
		std::cout << "\n";
		boost::compute::transform(y.begin(), y.end(), teach.begin(), y.begin(), boost::compute::minus<neu::scalar>());
		full_connected.init_delta(y);
		full_connected.update_delta_weight(input);
		full_connected.update_weight();
		neu::gpu_vector squared_errors(y.size());
		boost::compute::transform(y.begin(), y.end(), y.begin(), squared_errors.begin(), boost::compute::multiplies<neu::scalar>());
		auto error = boost::compute::accumulate(squared_errors.begin(), squared_errors.end(), 0.0f);
		std::cout << "here" << error << std::endl;
	}
	{
		auto weight = full_connected.get_weight();
		auto bias = full_connected.get_bias();
		std::cout << "weight"; print(weight); std::cout << "\n";
		std::cout << "bias"; print(bias); std::cout << "\n";
	}
}
*/
