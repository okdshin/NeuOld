#ifndef NEU_LAYER_ALGORITHM_HPP
#define NEU_LAYER_ALGORITHM_HPP
//20150604
#include <tuple>
#include <neu/tuple_algorithm.hpp>
#include <neu/basic_type.hpp>
#include <neu/layer_traits.hpp>
namespace neu {
	template<typename Layers, typename X>
	decltype(auto) feedforward_x(Layers&& layers, X const& x) {
		neu::layer_calc_u_and_y(neu::tuple_front(layers), x);
		neu::tuple_feedforward(layers, [](auto const& prev, auto& cur) {
			neu::layer_calc_u_and_y(cur, neu::layer_get_y(prev));
		});
	}

	template<typename Layers, typename InitialDelta>
	decltype(auto) feedback_delta(Layers& layers, InitialDelta const& initial_delta) {
		neu::layer_init_delta(neu::tuple_back(layers), initial_delta);
		neu::tuple_feedback(layers, [](auto& next, auto& cur) {
			neu::layer_calc_v(next);
			neu::layer_calc_delta(cur, neu::layer_get_v(next));
		});
	}

	template<typename Layers, typename X>
	decltype(auto) update_delta_weight(Layers& layers, X const& x) {
		neu::layer_update_delta_weight(neu::tuple_front(layers), x);
		neu::tuple_feedforward(layers, [](auto const& prev, auto& cur) {
			neu::layer_update_delta_weight(cur, neu::layer_get_y(prev));
		});
	}
	template<typename Layers>
	decltype(auto) update_weight(Layers& layers) {
		neu::layer_update_weight(neu::tuple_front(layers));
		neu::tuple_foreach(layers, [](auto& l) { neu::layer_update_weight(l); });
	}

	template<typename Layers, typename X>
	decltype(auto) feedforward_x_without_train(Layers&& layers, X const& x) {
		neu::layer_calc_y(neu::tuple_front(layers), x);
		neu::tuple_feedforward(layers, [](auto const& prev, auto& cur) {
			neu::layer_calc_y(cur, neu::layer_get_y(prev));
		});
	}
}// namespace neu

#endif //NEU_LAYER_ALGORITHM_HPP
