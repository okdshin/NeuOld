#ifndef NEU_SIGMOID_HPP
#define NEU_SIGMOID_HPP
//20150528
#include <cmath>
#include <neu/basic_type.hpp>
#include <neu/activate_func/differential.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, sigmoid_kernel, (float x), {
		return 1./(1.+exp(-x));
	});
	BOOST_COMPUTE_FUNCTION(float, diff_sigmoid_kernel, (float x), {
		const float sigma = 1./(1.+exp(-x));
		return sigma*(1.-sigma);
	});
	class sigmoid {
	public:
		decltype(auto) operator()(neu::gpu_vector x) const {
			boost::compute::transform(x.begin(), x.end(),
				x.begin(), neu::sigmoid_kernel);
			return x;
		}
		decltype(auto) operator()(neu::cpu_vector x) const {
			std::transform(x.begin(), x.end(), x.begin(),
				[](auto const& e){ return 1./(1.+std::exp(-e)); });
			return x;
		}
	};
	template<>
	class differential<sigmoid> {
	public:
		decltype(auto) operator()(neu::gpu_vector x) const {
			boost::compute::transform(x.begin(), x.end(),
				x.begin(), neu::diff_sigmoid_kernel);
			return x;
		}
		decltype(auto) operator()(neu::cpu_vector x) const {
			std::transform(x.begin(), x.end(), x.begin(),
				[](auto const& e){
					const neu::scalar sigma = 1./(1.+std::exp(-e));
					return sigma*(1.-sigma);
				});
			return x;
		}
	};
	using diff_sigmoid = differential<sigmoid>;
}// namespace neu

#endif //NEU_SIGMOID_HPP
