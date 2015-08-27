#include <neu/kernel.hpp>
#include <neu/vector_io.hpp>
#include <neu/activate_func/rectifier.hpp>
#include <neu/dropout_wrapper.hpp>

namespace neu_test {
	decltype(auto) dropout_wrapper_test() {
		//std::random_device rd;
		//std::mt19937 rand(rd());
		std::mt19937 rand(0); // fixed initial vector

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

		auto fc = neu::make_full_connected_layer(
			input_dim, output_dim, batch_size, neu::rectifier());
		fc.init_weight_randomly(
			[&rand, bin=std::uniform_real_distribution<>(0.f, 1.f)]() mutable {
				return bin(rand); });
		std::cout << "weight "; neu::print(fc.get_weight()); std::cout << std::endl;

		auto dfc = neu::make_dropout_wrapper(fc, 0.9f, rand);
		std::cout << "weight "; neu::print(dfc.get_layer().get_weight()); std::cout << std::endl;

		neu::layer_calc_u_and_y(dfc, input);
		auto y = neu::layer_get_y(dfc);
		std::cout << "y "; neu::print(y);
		neu::gpu_vector errors(y.size());
		boost::compute::transform(y.begin(), y.end(), teach.begin(), errors.begin(),
			boost::compute::minus<neu::scalar>());
		neu::layer_calc_delta(dfc, errors);
		neu::layer_update_delta_weight(dfc, input);
		neu::layer_update_weight(dfc);
		auto weight = neu::layer_get_weight(dfc.get_layer());
		std::cout << "updated weight"; neu::print(weight);
	}
}

#ifndef NEU_TEST_CPP_INCLUDE
#	define NEU_TEST_TEST_TARGET_FUNCTION neu_test::dropout_wrapper_test
#	include <test/include_main.hpp>
#endif
