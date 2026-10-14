# Add this bridge to an existing TensorIR build without patching upstream.
# The named project hook runs before upstream creates its targets; defer our
# target until those libraries and their transitive dependencies are defined.
set(TEMPEST_TENSOR_IR_RUNTIME_SOURCE "${CMAKE_CURRENT_LIST_DIR}")

function(tempest_add_tensor_ir_runtime)
  add_library(tempest_tensor_ir_runtime SHARED
    "${TEMPEST_TENSOR_IR_RUNTIME_SOURCE}/runtime.cpp")
  target_compile_features(tempest_tensor_ir_runtime PRIVATE cxx_std_17)
  target_include_directories(tempest_tensor_ir_runtime PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/include"
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})
  target_link_libraries(tempest_tensor_ir_runtime PRIVATE
    NVTensorIRCAPI MLIRCAPIIR)
  # LLVM builds without exceptions. Catch bridge-side allocation failures at
  # our C boundary while leaving the upstream libraries' flags unchanged.
  target_compile_options(tempest_tensor_ir_runtime PRIVATE -fexceptions)
  target_link_options(tempest_tensor_ir_runtime PRIVATE
    "LINKER:--exclude-libs,ALL")
  set_target_properties(tempest_tensor_ir_runtime PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tempest")
endfunction()

cmake_language(DEFER CALL tempest_add_tensor_ir_runtime)
