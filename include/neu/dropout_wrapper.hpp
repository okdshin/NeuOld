#ifndef NEU_DROPOUT_WRAPPER_HPP
#define NEU_DROPOUT_WRAPPER_HPP
//20150622
#include <neu/basic_type.hpp>
#include <neu/kernel.hpp>
#include <neu/layer_traits.hpp>
#include <neu/full_connected_layer.hpp>
namespace neu {
	const char dropout_train_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void dropout_train(
			const __global float* input, __global float* output,
			const __global float* dropout_mask, int inoutput_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);
			output[o+inoutput_dim*b] = dropout_mask[o]*input[o+inoutput_dim*b];
		}
	);
	const char dropout_test_kernel_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
		__kernel void dropout_test(
			const __global float* input, __global float* output,
			float p, int inoutput_dim)
		{
			const int b = get_global_id(1);
			const int o = get_global_id(0);
			output[o+inoutput_dim*b] = p*input[o+inoutput_dim*b];
		}
	);
	template<typename Layer, typename Rand>
	class dropout_wrapper {
	public:
		template<typename LayerArg, typename RandArg>
		dropout_wrapper(LayerArg&& layer, scalar p, RandArg&& rand,
			boost::compute::kernel const& dropout_train_kernel,
			boost::compute::kernel const& dropout_test_kernel)
			: layer_(std::forward<LayerArg>(layer)),
			p_(p), rand_(std::forward<RandArg>(rand)),
			dropout_train_kernel_(dropout_train_kernel),
			dropout_test_kernel_(dropout_test_kernel),
			output_dim_(layer_get_y_dim(layer_)),
			batch_size_(layer_get_batch_size(layer_)),
			dropout_mask_(output_dim_),
			y_(output_dim_*batch_size_),
			next_v_dash_(output_dim_*batch_size_),
			delta_dash_(output_dim_*batch_size_)
		{
			update_dropout_mask();
		}

		decltype(auto) get_y_dim() const { return layer_get_y_dim(layer_); }
		decltype(auto) get_batch_size() const {
			return layer_get_batch_size(layer_); }

		decltype(auto) calc_u_and_y(gpu_vector const& x) {
			layer_calc_u_and_y(layer_, x);
			auto y = layer_get_y(layer_);
			dropout_train_kernel_.set_args(layer_get_y(layer_), y_, dropout_mask_,
				static_cast<int>(output_dim_));
			std::size_t origin[] = {0, 0};
			std::size_t region[] = {output_dim_, batch_size_};
			boost::compute::system::default_queue().enqueue_nd_range_kernel(
				dropout_train_kernel_, 2, origin, region, nullptr).wait();
		}
		decltype(auto) get_y() const { return (y_); }
		decltype(auto) calc_y(gpu_vector const& x) {
			layer_calc_y(layer_, x);
			dropout_test_kernel_.set_args(layer_get_y(layer_), y_, p_,
				static_cast<int>(output_dim_));
			std::size_t origin[] = {0, 0};
			std::size_t region[] = {output_dim_, batch_size_};
			boost::compute::system::default_queue().enqueue_nd_range_kernel(
				dropout_test_kernel_, 2, origin, region, nullptr).wait();
		}

		decltype(auto) init_delta(gpu_vector const& delta) {
			assert(y_.size() == delta.size());
			dropout_train_kernel_.set_args(delta, delta_dash_, dropout_mask_,
				static_cast<int>(output_dim_));
			std::size_t origin[] = {0, 0};
			std::size_t region[] = {output_dim_, batch_size_};
			boost::compute::system::default_queue().enqueue_nd_range_kernel(
				dropout_train_kernel_, 2, origin, region, nullptr).wait();
			layer_init_delta(layer_, delta_dash_);
		}
		decltype(auto) calc_delta(gpu_vector const& next_v) {
			assert(y_.size() == next_v.size());
			dropout_train_kernel_.set_args(next_v, next_v_dash_, dropout_mask_,
				static_cast<int>(output_dim_));
			std::size_t origin[] = {0, 0};
			std::size_t region[] = {output_dim_, batch_size_};
			boost::compute::system::default_queue().enqueue_nd_range_kernel(
				dropout_train_kernel_, 2, origin, region, nullptr).wait();
			layer_calc_delta(layer_, next_v_dash_);
		}
		decltype(auto) calc_v() { layer_calc_v(layer_); }
		decltype(auto) get_v() const { return layer_get_v(layer_); }

		decltype(auto) update_delta_weight(gpu_vector const& x) {
			layer_update_delta_weight(layer_, x);
		}

		decltype(auto) update_weight() {
			layer_update_weight(layer_);
			update_dropout_mask();
		}

		decltype(auto) update_dropout_mask() {
			cpu_vector cpu_dropout_mask(dropout_mask_.size());
			std::generate(cpu_dropout_mask.begin(), cpu_dropout_mask.end(),
				[this, dist=std::uniform_real_distribution<>(0.f, 1.f)]() mutable {
					return static_cast<scalar>(dist(rand_) < p_ ? 1.f : 0.f);
				}
			);
			boost::compute::copy(cpu_dropout_mask.begin(), cpu_dropout_mask.end(),
				dropout_mask_.begin());
		}

		decltype(auto) get_dropout_mask() const { return (dropout_mask_); }

		decltype(auto) get_layer() { return (layer_); }

	private:
		Layer layer_;
		scalar p_;
		Rand rand_;

		boost::compute::kernel dropout_train_kernel_;
		boost::compute::kernel dropout_test_kernel_;

		std::size_t output_dim_;
		std::size_t batch_size_;

		gpu_vector dropout_mask_;
		gpu_vector y_;
		gpu_vector next_v_dash_;
		gpu_vector delta_dash_;
	};

	template<typename Layer, typename Rand>
	decltype(auto) make_dropout_wrapper(
			Layer&& layer, scalar p, Rand&& rand,
			boost::compute::kernel const& dropout_train_kernel
				=make_kernel(dropout_train_kernel_source, "dropout_train"),
			boost::compute::kernel const& dropout_test_kernel
				=make_kernel(dropout_test_kernel_source, "dropout_test")) {
		return dropout_wrapper<std::decay_t<Layer>, std::decay_t<Rand>>(
			std::forward<Layer>(layer), p, std::forward<Rand>(rand),
			dropout_train_kernel, dropout_test_kernel);
	}
}// namespace neu

#endif //NEU_DROPOUT_WRAPPER_HPP
