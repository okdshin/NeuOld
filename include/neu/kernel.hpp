#ifndef NEU_KERNEL_HPP
#define NEU_KERNEL_HPP
//20150625
#include <boost/compute/system.hpp>
#include <boost/compute/kernel.hpp>
namespace neu {
	using kernel = boost::compute::kernel;
	decltype(auto) make_kernel(const char* source, std::string const& name) {
		auto program = boost::compute::program::create_with_source(
			source, boost::compute::system::default_context());
		program.build();
		//TODO throw std::cout << program.build_log() << std::endl;
		return kernel(program, name.c_str());
	}
}// namespace neu

#endif //NEU_KERNEL_HPP
