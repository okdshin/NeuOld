add_executable( example_dropout_wrapper dropout_wrapper_learning_xor.cpp )
target_link_libraries( example_dropout_wrapper ${Boost_SYSTEM_LIBRARY} ${Boost_TIMER_LIBRARY} ${OpenCL_LIBRARIES})
set_target_properties( example_dropout_wrapper PROPERTIES OUTPUT_NAME "dropout_wrapper_lerning_xor" )
