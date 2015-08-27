#ifndef NEU_INDEX_RANGE_HPP
#define NEU_INDEX_RANGE_HPP
//20150604
namespace neu {
	template<class T>
	using invoke = typename T::type;

	template<typename T, T... Is>
	struct integer_sequence { using type = integer_sequence; };

	template<typename T, T First, std::make_signed_t<T> Step,
		std::make_unsigned_t<T> N, typename Enable=void>
	struct make_integer_range_dispatch;

	template<typename T, T First, std::make_signed_t<T> Step, std::make_unsigned_t<T> N>
	struct make_integer_range_dispatch<T, First, Step, N, std::enable_if_t<N==0>>
		: neu::integer_sequence<T> {};

	template<typename T, T First, std::make_signed_t<T> Step, std::make_unsigned_t<T> N>
	struct make_integer_range_dispatch<T, First, Step, N, std::enable_if_t<N==1>>
		: neu::integer_sequence<T, First> {};

	template<typename T, typename Seq, T Next>
	//template<typename T, typename Seq, std::make_signed_t<T> Next>
	struct make_integer_range_next_even {};
	template<typename T, T... Is, T Next>
	//template<typename T, T... Is, std::make_signed_t<T> Next>
	struct make_integer_range_next_even<T, neu::integer_sequence<T, Is...>, Next>
		: integer_sequence<T, Is..., (Is+Next)...> {};

	template<typename T, typename Seq, T Next, T Tail> //TODO check
	//template<typename T, typename Seq, std::make_signed_t<T> Next, T Tail>
	struct make_integer_range_next_odd {};
	template<typename T, T... Is, T Next, T Tail> //TODO check
	//template<typename T, T... Is, std::make_signed_t<T> Next, T Tail>
	struct make_integer_range_next_odd<T, neu::integer_sequence<T, Is...>, Next, Tail>
		: integer_sequence<T, Is..., (Is+Next)..., Tail> {};

	template<typename T, T First, std::make_signed_t<T> Step, std::make_unsigned_t<T> N>
	struct make_integer_range_dispatch<T, First, Step, N,
		std::enable_if_t<(N>1 && N%2==0)>>
		: neu::make_integer_range_next_even<T,
			neu::invoke<neu::make_integer_range_dispatch<T, First, Step, N/2>>,
			(N/2)*Step> {};

	template<typename T, T First, std::make_signed_t<T> Step, std::make_unsigned_t<T> N>
	struct make_integer_range_dispatch<T, First, Step, N,
		std::enable_if_t<(N>1 && N%2==1)>>
		: neu::make_integer_range_next_odd<T,
			neu::invoke<neu::make_integer_range_dispatch<T, First, Step, N/2>>,
			(N/2)*Step, First+(N-1)*Step> {};

	template<typename T, T First, T Last, std::make_signed_t<T> Step>
	struct make_integer_range_impl
		: neu::make_integer_range_dispatch<T, First, Step,
			(static_cast<std::make_signed_t<T>>(Last-First)
			 	+(Step>0? Step-1 : Step+1))/Step> {};

	template<typename T, T First, T Last, std::make_signed_t<T> Step>
	struct make_integer_range_aux
		: neu::make_integer_range_impl<T, First, Last, Step>
	{
		static_assert(
			(First<Last && Step > 0) || (First > Last && Step < 0) || (First == Last),
			"(First<Last && Step > 0) || (First > Last && Step < 0) || (First == Last)");
	};
	
	template<typename T, T First, T Last,
		std::make_signed_t<T> Step=(Last>First ? 1 : -1)>
	using make_integer_range =
		neu::invoke<neu::make_integer_range_aux<T, First, Last, Step>>;
	
	template<std::size_t First, std::size_t Last,
		std::make_signed_t<std::size_t> Step=(Last>First ? 1 : -1)>
	using make_index_range =
		neu::invoke<neu::make_integer_range_aux<std::size_t, First, Last, Step>>;
	template<std::size_t... Is>
	using index_range = neu::integer_sequence<std::size_t, Is...>;

}// namespace neu

#endif //NEU_INDEX_RANGE_HPP
