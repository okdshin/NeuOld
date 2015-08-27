#ifndef NEU_TEST_INCLUDE_MAIN_HPP
#define NEU_TEST_INCLUDE_MAIN_HPP

#ifndef NEU_TEST_CPP_INCLUDE

#ifndef NEU_TEST_TEST_TARGET_FUNCTION
#	error undefined NEU_TEST_TEST_TARGET_FUNCTION
#endif

#include <iostream>

#define NEU_TEST_PP_STRINGIZE(x) NEU_TEST_PP_STRINGIZE_AUX(x)
#define NEU_TEST_PP_STRINGIZE_AUX(x) #x

int main() {
	std::cout << "neu test execute(" << NEU_TEST_PP_STRINGIZE(NEU_TEST_TEST_TARGET_FUNCTION) << "):" << std::endl;
	try {
		NEU_TEST_TEST_TARGET_FUNCTION();
	}
	catch(...) {
		std::cout << "  testneu failed for unknown error." << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "  neu test succeeded." << std::endl;
}

#endif //NEU_TEST_CPP_INCLUDE

#endif // #ifndef NEU_TEST_INCLUDE_MAIN_HPP
