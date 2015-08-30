#ifndef NEU_FULL_CONNECTED_LAYER_HPP
#define NEU_FULL_CONNECTED_LAYER_HPP
//20150622
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/learning_rate_gen/fixed_learning_rate_gen.hpp>
namespace neu {
	const char multiply_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void multiply(
			const __global float* input, __global float* output,
			const __global float* weight, const __global float* bias,
			const int input_dim, const int output_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			float sum = bias[o];
			for(int i = 0; i < input_dim; ++i) {
				sum += weight[i+input_dim*o]*input[i+input_dim*b];
			}
			output[o+output_dim*b] = sum;
		}
	);
	const char multiply_back_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void multiply_back(
			const __global float* input, __global float* output,
			const __global float* weight,
			const int input_dim, const int output_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);

			float sum = 0.0;
			for(int i = 0; i < input_dim; ++i) {
				sum += weight[o+output_dim*i]*input[i+input_dim*b];
			}
			output[o+output_dim*b] = sum;
		}
	);
	const char update_delta_weight_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void update_delta_weight(
			const __global float* input, const __global float* delta,
			__global float* delta_weight, __global float* delta_bias,
			const int input_dim, const int output_dim, const int batch_size)
		{
			const int gr = get_global_id(1);
			const int gc = get_global_id(0);

			float weight_sum = 0.0;
			float bias_sum = 0.0;
			for(int b = 0; b < batch_size; ++b) {
				weight_sum += delta[gr+output_dim*b]*input[gc+input_dim*b];
				bias_sum += delta[gr+output_dim*b];
			}
			delta_weight[gc+input_dim*gr] += weight_sum/batch_size;
			delta_bias[gr] += bias_sum/batch_size;
		}
	);
	template<typename ActivateFunc, typename DiffActivateFunc,
		typename LearningRateGen>
	class full_connected_layer {
	public:
		full_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			ActivateFunc const& activate_func, DiffActivateFunc const& diff_activate_func,
			LearningRateGen const& learning_rate_gen,
			boost::compute::kernel const& multiply_kernel,
			boost::compute::kernel const& multiply_back_kernel,
			boost::compute::kernel const& update_delta_weight_kernel)
			: input_dim_{input_dim}, output_dim_{output_dim}, batch_size_{batch_size},
			activate_func_(activate_func), diff_activate_func_(diff_activate_func),
			learning_rate_gen_(learning_rate_gen),
			multiply_kernel_(multiply_kernel),
			multiply_back_kernel_(multiply_back_kernel),
			update_delta_weight_kernel_(update_delta_weight_kernel),
			weight_(input_dim_*output_dim_), bias_(output_dim_),
			u_(output_dim_*batch_size_),
			y_(output_dim_*batch_size_),
			delta_(output_dim_*batch_size_),
			v_(input_dim_*batch_size_),
			delta_weight_(input_dim_*output_dim_), delta_bias_(output_dim_)
		{
			init_delta_weight();
		}

		decltype(auto) init_delta_weight() {
			boost::compute::fill(delta_weight_.begin(), delta_weight_.end(), 0.f);
			boost::compute::fill(delta_bias_.begin(), delta_bias_.end(), 0.f);
		}

		template<typename Rand>
		decltype(auto) init_weight_randomly(Rand const& rand) {
			cpu_vector cpu_weight(weight_.size());
			std::generate(cpu_weight.begin(), cpu_weight.end(), rand);
			boost::compute::copy(cpu_weight.begin(), cpu_weight.end(), weight_.begin());

			cpu_vector cpu_bias(bias_.size());
			std::generate(cpu_bias.begin(), cpu_bias.end(), rand);
			boost::compute::copy(cpu_bias.begin(), cpu_bias.end(), bias_.begin());
		}

		decltype(auto) get_weight() const { return weight_; }
		decltype(auto) get_bias() const { return bias_; }

		decltype(auto) get_u() const { return (u_); }

		decltype(auto) get_y_dim() const { return output_dim_; }
		decltype(auto) get_batch_size() const { return batch_size_; }

		decltype(auto) calc_u_and_y(gpu_vector const& x) {
			neu::execute_nd_range_kernel<2>(
				multiply_kernel_, {0, 0}, {output_dim_, batch_size_},
				x, u_, weight_, bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_)).wait();
			y_ = activate_func_(u_);
		}
		decltype(auto) calc_y(gpu_vector const& x) { calc_u_and_y(x); }
		decltype(auto) get_y() const { return (y_); }

		decltype(auto) init_delta(gpu_vector const& delta) { delta_ = delta; }
		decltype(auto) calc_delta(gpu_vector const& next_v) {
			assert(u_.size() == next_v.size());
			auto df = diff_activate_func_(u_);

			// delta[l] = df * v[l+1]
			boost::compute::transform(df.begin(), df.end(), next_v.begin(),
				delta_.begin(), boost::compute::multiplies<scalar>());
		}

		decltype(auto) calc_v() {
			neu::execute_nd_range_kernel<2>(
				multiply_back_kernel_, {0, 0}, {input_dim_, batch_size_},
				delta_, v_, weight_,
				static_cast<int>(output_dim_), static_cast<int>(input_dim_)).wait();
		}
		decltype(auto) get_v() const { return (v_); }

		decltype(auto) update_delta_weight(gpu_vector const& x) {
			neu::execute_nd_range_kernel<2>(
				update_delta_weight_kernel_, {0, 0}, {input_dim_, output_dim_},
				x, delta_, delta_weight_, delta_bias_,
				static_cast<int>(input_dim_), static_cast<int>(output_dim_),
				static_cast<int>(batch_size_)).wait();
		}

		//TODO customizable
		decltype(auto) update_weight() {
			/*
			// weight -= delta_weight
			boost::compute::transform(weight_.begin(), weight_.end(),
				delta_weight_.begin(), weight_.begin(), boost::compute::minus<scalar>());

			// bias -= delta_bias
			boost::compute::transform(bias_.begin(), bias_.end(),
				delta_bias_.begin(), bias_.begin(), boost::compute::minus<scalar>());
			*/

			learning_rate_gen_(weight_, bias_, delta_weight_, delta_bias_);
			init_delta_weight();
		}

	private:
		std::size_t input_dim_;
		std::size_t output_dim_;
		std::size_t batch_size_;

		ActivateFunc activate_func_;
		DiffActivateFunc diff_activate_func_;
		LearningRateGen learning_rate_gen_;
		boost::compute::kernel multiply_kernel_;
		boost::compute::kernel multiply_back_kernel_;
		boost::compute::kernel update_delta_weight_kernel_;

		gpu_vector weight_;
		gpu_vector bias_;
		gpu_vector u_;
		gpu_vector y_;
		gpu_vector delta_;
		gpu_vector v_;
		gpu_vector delta_weight_;
		gpu_vector delta_bias_;
	};

	template<typename ActivateFunc, typename LearningRateGen=fixed_learning_rate_gen>
	decltype(auto) make_full_connected_layer(
			std::size_t input_dim, std::size_t output_dim, std::size_t batch_size,
			ActivateFunc const& activate_func,
			LearningRateGen const& learning_rate_gen=fixed_learning_rate_gen(0.1f),
			boost::compute::kernel const& multiply_kernel
				=make_kernel(multiply_kernel_source, "multiply"),
			boost::compute::kernel const& multiply_back_kernel
				=make_kernel(neu::multiply_back_kernel_source, "multiply_back"),
			boost::compute::kernel const& update_delta_weight_kernel
				=make_kernel(update_delta_weight_kernel_source, "update_delta_weight")) {
		static_assert(std::is_same<LearningRateGen, fixed_learning_rate_gen>::value, "");
		return full_connected_layer<ActivateFunc, differential<ActivateFunc>,
				LearningRateGen>(
			input_dim, output_dim, batch_size,
			activate_func, differential<ActivateFunc>(), learning_rate_gen,
			multiply_kernel, multiply_back_kernel, update_delta_weight_kernel);
	}
}// namespace neu

#endif //NEU_FULL_CONNECTED_LAYER_HPP
