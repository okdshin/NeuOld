add_executable( example_full_connected_layer_leaning_xor_adagrad full_connected_layer_learning_xor_adagrad.cpp )
target_link_libraries( example_full_connected_layer_leaning_xor_adagrad ${OpenCL_LIBRARIES})
set_target_properties( example_full_connected_layer_leaning_xor_adagrad PROPERTIES OUTPUT_NAME "full_connected_layer_lerning_xor_adagrad" )
