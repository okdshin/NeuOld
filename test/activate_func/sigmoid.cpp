#include <neu/activate_func/sigmoid.hpp>
#include <neu/vector_io.hpp>

namespace neu_test {
	decltype(auto) activate_func_sigmoid_test() {
		neu::sigmoid a;
		neu::differential<neu::sigmoid> da;
		const auto cpu_x = neu::cpu_vector{0.f, -0.1f, 0.1f, -100.f, 100.f};
		const auto gpu_x = neu::to_gpu_vector(cpu_x);
		{
			auto y = neu::to_cpu_vector(a(gpu_x));
			//assert((y == neu::cpu_vector{0.f, 0.f, 0.1f, 0.f, 100.f}));
			neu::print(y); //TODO
		}
		{
			auto dy = neu::to_cpu_vector(da(gpu_x));
			//assert((dy == neu::cpu_vector{0.f, 0.f, 1.f, 0.f, 1.f}));
			neu::print(dy); //TODO
		}
	}
}

#ifndef NEU_TEST_CPP_INCLUDE
#	define NEU_TEST_TEST_TARGET_FUNCTION neu_test::activate_func_sigmoid_test
#	include <test/include_main.hpp>
#endif

