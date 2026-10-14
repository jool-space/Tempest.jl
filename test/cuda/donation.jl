@testset "Grouped CUDA updates across independent arrays" begin
    arrays = ntuple(i -> fill(Float32(i) / 100, (17, 63, 127, 513, 1025)[mod1(i, 5)]), 100)
    update(xs) = (map(x -> x .* 0.9f0 .+ 1.0f0, xs),)
    expected = arrays
    ex = prepare_forward(update, map(CuArray, arrays); donate = (1,))
    @test length(ex.calls) == 1
    @test isempty(ex.buffers)
    @test isempty(ex.state.commits)
    pointers = map(pointer, ex.state.buffers)
    capture!(ex)
    @test map(Array, only(outputs(ex))) == arrays
    for iteration = 1:3
        expected = only(update(expected))
        actual = only(replay!(ex))
        for (a, b) in zip(actual, expected)
            @test Array(a) ≈ b
        end
        @test map(pointer, ex.state.buffers) == pointers
        if iteration == 2
            saved = map(Array, actual)
            capture!(ex)
            @test map(Array, only(outputs(ex))) == saved
        end
    end
    bind_argument!(ex, 1, arrays)
    for (a, b) in zip(only(replay!(ex)), only(update(arrays)))
        @test Array(a) ≈ b
    end
    staged = prepare_forward(
        update,
        map(CuArray, arrays);
        donate = (1,),
        inplace = false,
        grouped = false,
    )
    capture!(staged)
    @test length(staged.calls) == 100
    @test length(staged.state.commits) == 100
    for (a, b) in zip(only(replay!(staged)), only(update(arrays)))
        @test Array(a) ≈ b
    end
end

@testset "CUDA rejects racing in-place maps" begin
    input = Float32[1, 2, 3, 4]
    update(x) = (x[[4, 3, 2, 1]] .+ 1.0f0,)
    ex = prepare_forward(update, CuArray(input); donate = (1,))
    @test isempty(ex.storage.donations)
    capture!(ex)
    @test Array(only(replay!(ex))) == reverse(input) .+ 1
    @test Array(only(replay!(ex))) == input .+ 2
end
