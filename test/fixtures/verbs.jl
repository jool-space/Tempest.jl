# A public mathematical function, plus its symbolic boundary. No allocation
# wrapper or implementation keyword is required at the model call site.
pair_wave(x) = reference(pair_wave, x)
pair_wave(x::Tempest.TracedArray) = record_call(pair_wave, x)
Tempest.output_spec(::typeof(pair_wave), x) = (; s = TensorSpec(x), c = TensorSpec(x))
Tempest.reference(::typeof(pair_wave), x) = (; s = sin.(x), c = cos.(x))
wave_back(gs, gc, s, c) = reference(wave_back, gs, gc, s, c)
wave_back(gs::Tempest.TracedArray, gc, s, c) = record_call(wave_back, gs, gc, s, c)
Tempest.output_spec(::typeof(wave_back), gs, gc, s, c) = TensorSpec(gs)
Tempest.reference(::typeof(wave_back), gs, gc, s, c) = gs .* c .- gc .* s
Tempest.verb_pullback(::typeof(pair_wave), xs, y, g) = (wave_back(g.s, g.c, y.s, y.c),)

struct WaveImplementation <: Implementation
    mode::Symbol
end

Tempest.implementation_name(::WaveImplementation) = :wave_kernel

function Tempest.support(c::WaveImplementation, r::CallRequest)
    if !(r.call.f in (pair_wave, wave_back))
        return Unsupported("not a wave operation")
    end
    if c.mode === :reject
        return Unsupported("test semantic rejection")
    end
    if !(r.target isa CPUTarget)
        return Unsupported("CPU implementation")
    end
    Supported()
end

function Tempest.build_implementation(
    c::WaveImplementation,
    r::CallRequest,
    runtime,
    out,
    inputs,
)
    if c.mode === :unavailable
        return Unsupported("test semantic engine unavailable")
    end
    if c.mode === :broken
        error("test semantic compiler failure")
    end
    launch = if r.call.f === pair_wave
        () -> begin
            x = only(inputs)
            for i in eachindex(x)
                out.s[i], out.c[i] = sincos(x[i])
            end
        end
    else
        () -> begin
            gs, gc, s, c = inputs
            for i in eachindex(out)
                out[i] = gs[i] * c[i] - gc[i] * s[i]
            end
        end
    end
    BuiltImplementation(launch, (out, inputs), 0)
end
