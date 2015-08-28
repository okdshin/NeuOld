#ifndef NEU_RECTIFIER_HPP
#define NEU_RECTIFIER_HPP
//20150528
#include <neu/basic_type.hpp>
#include <neu/activate_func/differential.hpp>
namespace neu {
	BOOST_COMPUTE_FUNCTION(float, rectifier_kernel, (float x), {
		return x > 0 ? x : 0;
	});
	BOOST_COMPUTE_FUNCTION(float, diff_rectifier_kernel, (float x), {
		return x > 0 ? 1 : 0;
	});
	class rectifier {
	public:
		decltype(auto) operator()(neu::gpu_vector const& x) const {
			neu::gpu_vector result(x.size());
			boost::compute::transform(x.begin(), x.end(),
				result.begin(), neu::rectifier_kernel);
			return result;
		}
	};
	template<>
	class differential<rectifier> {
	public:
		decltype(auto) operator()(neu::gpu_vector const& x) const {
			neu::gpu_vector result(x.size());
			boost::compute::transform(x.begin(), x.end(),
				result.begin(), neu::diff_rectifier_kernel);
			return result;
		}
	};
}// namespace neu

#endif //NEU_RECTIFIER_HPP
