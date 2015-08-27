add_executable( test_activate_func_test_identity identity.cpp )
target_link_libraries( test_activate_func_test_identity ${OpenCL_LIBRARIES})
set_target_properties( test_activate_func_test_identity PROPERTIES OUTPUT_NAME "identity" )
add_test( test_activate_func_test_identity identity )

add_executable( test_activate_func_test_rectifier rectifier.cpp )
target_link_libraries( test_activate_func_test_rectifier  ${OpenCL_LIBRARIES})
set_target_properties( test_activate_func_test_rectifier PROPERTIES OUTPUT_NAME "rectifier" )
add_test( test_activate_func_test_rectifier rectifier )
