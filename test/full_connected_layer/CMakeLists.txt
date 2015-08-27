add_executable( test_full_connected_layer full_connected_layer.cpp )
target_link_libraries( test_full_connected_layer ${OpenCL_LIBRARIES})
set_target_properties( test_full_connected_layer PROPERTIES OUTPUT_NAME "full_connected_layer" )
add_test( test_full_connected_layer full_connected_layer)
