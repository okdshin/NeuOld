#include <neu/activate_func/identity.hpp>

namespace testneu {
	decltype(auto) activate_func_identity_test() {
		neu::identity a;
		neu::differential<neu::identity> da;
		const auto cpu_x = neu::cpu_vector{0.f, -0.1f, 0.1f, -100.f, 100.f};
		const auto gpu_x = neu::to_gpu_vector(cpu_x);
		{
			auto y = neu::to_cpu_vector(a(gpu_x));
			assert((y == cpu_x));
		}
		{
			auto dy = neu::to_cpu_vector(da(gpu_x));
			assert((dy == neu::cpu_vector(5, 1.f)));
		}
	}
}

#ifndef NEU_TEST_CPP_INCLUDE
#	define NEU_TEST_TEST_TARGET_FUNCTION testneu::activate_func_identity_test
#	include <test/include_main.hpp>
#endif

