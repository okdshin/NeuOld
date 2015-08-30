#ifndef NEU_BASIC_TYPE_HPP
#define NEU_BASIC_TYPE_HPP
//20150619
#include<vector>
#include<boost/compute/container/vector.hpp>
#include<boost/compute/algorithm.hpp>
namespace neu {
	using scalar = float;
	using cpu_vector = std::vector<neu::scalar>;
	using gpu_vector = boost::compute::vector<neu::scalar>;
	using gpu_indices = boost::compute::vector<int>;

	template<typename Array>
	decltype(auto) to_gpu_vector(Array const& a) {
		using std::begin;
		using std::end;
		return neu::gpu_vector(begin(a), end(a));
	}

	template<typename T>
	decltype(auto) to_cpu_vector(boost::compute::vector<T> const& x) {
		std::vector<T> cpu_x(x.size());
		boost::compute::copy(x.begin(), x.end(), cpu_x.begin());
		return cpu_x;
	}

}// namespace neu

#endif //NEU_BASIC_TYPE_HPP
