#ifndef NEU_VECTOR_IO_HPP
#define NEU_VECTOR_IO_HPP
//20150828
#include <iostream>
#include<neu/basic_type.hpp>
namespace neu {
	decltype(auto) print(gpu_vector const& vec) {
		std::cout << "[";
		boost::compute::copy(vec.begin(), vec.end(),
			std::ostream_iterator<scalar>(std::cout, ", "));
		std::cout << "]" << std::endl;
	}
}// namespace neu

#endif //NEU_VECTOR_IO_HPP
