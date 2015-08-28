#ifndef NEU_IDENTITY_HPP
#define NEU_IDENTITY_HPP
//20150528
#include <neu/basic_type.hpp>
#include <neu/activate_func/differential.hpp>
namespace neu {
	class identity {
	public:
		template<typename T>
		decltype(auto) operator()(T&& x) const {
			return std::forward<T>(x);
		}
	};
	template<>
	class differential<identity> {
	public:
		decltype(auto) operator()(neu::gpu_vector x) const {
			boost::compute::fill(x.begin(), x.end(), 1.);
			return x;
		}
	};
}// namespace neu

#endif //NEU_IDENTITY_HPP
