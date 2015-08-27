set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin)

add_executable( test_activate_func_test_identity identity.cpp )
target_link_libraries( test_activate_func_test_identity ${OpenCL_LIBRARIES})
set_target_properties( test_activate_func_test_identity PROPERTIES OUTPUT_NAME "identity" )
add_test( test_activate_func_test_identity identity )

add_executable( test_activate_func_test_rectifier rectifier.cpp )
target_link_libraries( test_activate_func_test_rectifier  ${OpenCL_LIBRARIES})
set_target_properties( test_activate_func_test_rectifier PROPERTIES OUTPUT_NAME "rectifier" )
add_test( test_activate_func_test_rectifier rectifier )

add_executable( test_dropout_wrapper dropout_wrapper.cpp )
target_link_libraries( test_dropout_wrapper ${OpenCL_LIBRARIES})
set_target_properties( test_dropout_wrapper PROPERTIES OUTPUT_NAME "dropout_wrapper" )
add_test( test_dropout_wrapper dropout_wrapper )
