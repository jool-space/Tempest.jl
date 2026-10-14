// Research control: NVIDIA TensorIR v0.1.4, column-major Float32.
// Width 256 passed cubin compilation and reference verification on GB10.
nv_tensor_ir.graph @centered_layernorm_256(
    %x: tensor<256x7xf32> {nv_tensor_ir.stride = "(1,256)"})
    -> (tensor<256x7xf32> {nv_tensor_ir.stride = "(1,256)"},
        tensor<1x7xf32> {nv_tensor_ir.stride = "(1,1)"},
        tensor<1x7xf32> {nv_tensor_ir.stride = "(1,1)"})
    attributes {tile_size = array<i32: 256, 4>} {
  %count = constant dense<256.0> : tensor<1x7xf32>
  %epsilon = constant dense<1.0e-5> : tensor<1x7xf32>
  %total = reduce(%x) <dimensions = [0], reduction_mode = <add>>
    : tensor<256x7xf32> -> tensor<1x7xf32>
  %mean = div %total, %count : tensor<1x7xf32>
  %mean_wide = broadcast %mean : tensor<1x7xf32> -> tensor<256x7xf32>
  %centered = sub %x, %mean_wide : tensor<256x7xf32>
  %squared = mul %centered, %centered : tensor<256x7xf32>
  %square_total = reduce(%squared) <dimensions = [0], reduction_mode = <add>>
    : tensor<256x7xf32> -> tensor<1x7xf32>
  %variance = div %square_total, %count : tensor<1x7xf32>
  %stabilized = add %variance, %epsilon : tensor<1x7xf32>
  %rstd = rsqrt %stabilized : tensor<1x7xf32>
  %rstd_wide = broadcast %rstd : tensor<1x7xf32> -> tensor<256x7xf32>
  %y = mul %centered, %rstd_wide : tensor<256x7xf32>
  results %y, %mean, %rstd : tensor<256x7xf32>, tensor<1x7xf32>, tensor<1x7xf32>
}
