add_executable( test_dropout_wrapper dropout_wrapper.cpp )
target_link_libraries( test_dropout_wrapper ${OpenCL_LIBRARIES})
set_target_properties( test_dropout_wrapper PROPERTIES OUTPUT_NAME "dropout_wrapper" )
add_test( test_dropout_wrapper dropout_wrapper )
