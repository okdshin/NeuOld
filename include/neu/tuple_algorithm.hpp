#ifndef NEU_TUPLE_ALGORITHM_HPP
#define NEU_TUPLE_ALGORITHM_HPP
//20150604
#include <tuple>
#include <neu/index_range.hpp>
namespace neu {
	template<typename Tuple>
	decltype(auto) tuple_front(Tuple&& t) {
		return std::get<0>(t);
	}
	template<typename Tuple>
	decltype(auto) tuple_back(Tuple&& t) {
		return std::get<std::tuple_size<std::remove_reference_t<Tuple>>::value-1>(t);
	}
	template<typename Tuple, typename F, std::size_t... Is>
	decltype(auto) tuple_foreach_impl(Tuple&& tuple, F&& f, neu::index_range<Is...>) {
		volatile auto l = {(f(std::get<Is>(std::forward<Tuple>(tuple))), nullptr)...};
		static_cast<void>(l); // refusing unused value and expression result warning
	}
	template<typename Tuple, typename F>
	decltype(auto) tuple_foreach(Tuple&& tuple, F&& f) {
		tuple_foreach_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
			neu::make_index_range<
				0, std::tuple_size<std::remove_reference_t<Tuple>>::value>());
	}

	template<std::make_signed_t<std::size_t> Dir,
		typename Tuple, typename F, std::size_t... Is>
	decltype(auto) tuple_feed_impl(Tuple&& tuple, F&& f, 
			neu::index_range<Is...>) {
		volatile auto l = {(f(std::get<Is>(tuple), std::get<Is+Dir>(tuple)), nullptr)...};
		static_cast<void>(l); // refusing unused value and expression result warning
	}
	template<typename Tuple, typename F>
	decltype(auto) tuple_feedforward(Tuple&& tuple, F&& f) {
		neu::tuple_feed_impl<+1>(std::forward<Tuple>(tuple), std::forward<F>(f),
			neu::make_index_range<
				0, std::tuple_size<std::remove_reference_t<Tuple>>::value-1>());
	}
	template<typename Tuple, typename F>
	decltype(auto) tuple_feedback(Tuple&& tuple, F&& f) {
		neu::tuple_feed_impl<-1>(std::forward<Tuple>(tuple), std::forward<F>(f),
			neu::make_index_range<
				std::tuple_size<std::remove_reference_t<Tuple>>::value-1, 0>());
	}
}// namespace neu

#endif //NEU_TUPLE_ALGORITHM_HPP
