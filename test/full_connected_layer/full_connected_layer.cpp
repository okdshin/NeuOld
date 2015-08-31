//TODO
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/activate_func/identity.hpp>
#include <neu/full_connected_layer.hpp>
#include <neu/vector_io.hpp>

namespace neu_test {
	decltype(auto) full_connected_layer_test() {
		using namespace neu;
		{
			auto batch_size = 2u;
			auto input_dim = 3u;
			auto output_dim = 4u;
			auto input_vector = to_gpu_vector(cpu_vector{0,1,2, 3,4,5});
			auto weight = to_gpu_vector(cpu_vector{0,1,2, 3,4,5, 6,7,8, 9,10,11});
			auto bias = to_gpu_vector(cpu_vector{0, 1, 2, 3});
			gpu_vector output_vector(output_dim*batch_size);
			execute_nd_range_kernel<2>(
				make_kernel(multiply_kernel_source, "multiply"),
				{0, 0}, {output_dim, batch_size},
				input_vector, output_vector, weight, bias,
				static_cast<int>(input_dim), static_cast<int>(output_dim));
			assert("full_connected_layer_test_multiply" &&
				(to_cpu_vector(output_vector) == cpu_vector{5,15,25,35, 14,51,88,125}));
		}
		{
			auto batch_size = 2u;
			auto input_dim = 3u;
			auto output_dim = 4u;
			auto delta = to_gpu_vector(cpu_vector{0,1,2,3, 4,5,6,7});
			auto weight = to_gpu_vector(cpu_vector{0,1,2, 3,4,5, 6,7,8, 9,10,11});
			gpu_vector v(input_dim*batch_size);
			neu::execute_nd_range_kernel<2>(
				make_kernel(multiply_back_kernel_source, "multiply_back"),
				{0, 0}, {input_dim, batch_size},
				delta, v, weight,
				static_cast<int>(output_dim), static_cast<int>(input_dim));
			assert("full_connected_layer_test_multiply_back" &&
				(to_cpu_vector(v) == cpu_vector{42,48,54, 114,136,158}));
		}
		{
			auto batch_size = 2u;
			auto input_dim = 3u;
			auto output_dim = 4u;
			auto input_vector = to_gpu_vector(cpu_vector{0,1,2, 3,4,5});
			auto delta = to_gpu_vector(cpu_vector{0,1,2,3, 4,5,6,7});
			auto delta_weight = to_gpu_vector(cpu_vector{0,1,2, 3,4,5, 6,7,8, 9,10,11});
			auto delta_bias = to_gpu_vector(cpu_vector{0,1,2,3});
			neu::execute_nd_range_kernel<2>(
				make_kernel(update_delta_weight_kernel_source, "update_delta_weight"),
				{0, 0}, {input_dim, output_dim},
				input_vector, delta, delta_weight, delta_bias,
				static_cast<int>(input_dim), static_cast<int>(output_dim),
				static_cast<int>(batch_size));
			assert("full_connected_layer_test_update_delta_weight_weight" &&
				(to_cpu_vector(delta_weight) == cpu_vector{
				 	6,9,12, 10.5,14.5,18.5, 15,20,25, 19.5,25.5,31.5}));
			assert("full_connected_layer_test_update_delta_weight_weight_bias" &&
				(to_cpu_vector(delta_bias) == cpu_vector{2,4,6,8}));
		}
	}
}

#ifndef NEU_TEST_CPP_INCLUDE
#	define NEU_TEST_TEST_TARGET_FUNCTION neu_test::full_connected_layer_test
#	include <test/include_main.hpp>
#endif
