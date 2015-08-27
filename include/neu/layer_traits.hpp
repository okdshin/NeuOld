#ifndef NEU_LAYER_TRAITS_HPP
#define NEU_LAYER_TRAITS_HPP
//20150604

namespace neu {
	namespace layer_traits {
		template<typename L>
		class get_y_dim {
		public:
			static decltype(auto) call(L const& l) { return l.get_y_dim(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_get_y_dim(L const& l) {
		return neu::layer_traits::get_y_dim<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class get_batch_size {
		public:
			static decltype(auto) call(L const& l) { return l.get_batch_size(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_get_batch_size(L const& l) {
		return neu::layer_traits::get_batch_size<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class get_y {
		public:
			static decltype(auto) call(L const& l) { return l.get_y(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_get_y(L const& l) {
		return neu::layer_traits::get_y<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class calc_u_and_y {
		public:
			template<typename X>
			static decltype(auto) call(L& l, X const& x) { l.calc_u_and_y(x); }
		};
	}// namespace layer_traits
	template<typename L, typename X>
	decltype(auto) layer_calc_u_and_y(L& l, X const& x) {
		neu::layer_traits::calc_u_and_y<L>::call(l, x);
	}

	namespace layer_traits {
		template<typename L>
		class calc_y {
		public:
			template<typename X>
			static decltype(auto) call(L& l, X const& x) { l.calc_y(x); }
		};
	}// namespace layer_traits
	template<typename L, typename X>
	decltype(auto) layer_calc_y(L& l, X const& x) {
		neu::layer_traits::calc_y<L>::call(l, x);
	}

	namespace layer_traits {
		template<typename L>
		class update_weight {
		public:
			static decltype(auto) call(L& l) { l.update_weight(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_update_weight(L& l) {
		neu::layer_traits::update_weight<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class update_delta_weight {
		public:
			template<typename X>
			static decltype(auto) call(L& l, X const& x) { l.update_delta_weight(x); }
		};
	}// namespace layer_traits
	template<typename L, typename X>
	decltype(auto) layer_update_delta_weight(L& l, X const& x) {
		neu::layer_traits::update_delta_weight<L>::call(l, x);
	}

	namespace layer_traits {
		template<typename L>
		class init_delta {
		public:
			template<typename Delta>
			static decltype(auto) call(L& l, Delta const& delta) { l.init_delta(delta); }
		};
	}// namespace layer_traits
	template<typename L, typename Delta>
	decltype(auto) layer_init_delta(L& l, Delta const& delta) {
		neu::layer_traits::init_delta<L>::call(l, delta);
	}

	namespace layer_traits {
		template<typename L>
		class calc_delta {
		public:
			template<typename V>
			static decltype(auto) call(L& l, V const& v) { l.calc_delta(v); }
		};
	}// namespace layer_traits
	template<typename L, typename V>
	decltype(auto) layer_calc_delta(L& l, V const& v) {
		neu::layer_traits::calc_delta<L>::call(l, v);
	}

	namespace layer_traits {
		template<typename L>
		class get_delta {
		public:
			static decltype(auto) call(L const& l) { return l.get_delta(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_get_delta(L const& l) {
		return neu::layer_traits::get_delta<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class calc_v {
		public:
			static decltype(auto) call(L& l) { l.calc_v(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_calc_v(L& l) {
		neu::layer_traits::calc_v<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class get_v {
		public:
			static decltype(auto) call(L const& l) { return l.get_v(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_get_v(L& l) {
		return neu::layer_traits::get_v<L>::call(l);
	}

	namespace layer_traits {
		template<typename L>
		class get_weight {
		public:
			static decltype(auto) call(L const& l) { return l.get_weight(); }
		};
	}// namespace layer_traits
	template<typename L>
	decltype(auto) layer_get_weight(L const& l) {
		return neu::layer_traits::get_weight<L>::call(l);
	}
}// namespace neu

#endif //NEU_LAYER_TRAITS_HPP
