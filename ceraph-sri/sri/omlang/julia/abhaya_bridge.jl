# Abhaya bridge. Binary Float32 path, plus the @sphota launch schedule.
#
# mode=damp: input = [theta..., xi...] (2N), extra arg lam, output = A (N)
# The block ladder must match sri.omlang.abhaya.sphota_schedule.

length(ARGS) >= 1 || error("Expected mode or --schedule")

function sphota_schedule(n::Int, density::Float64)
    block = if density < 0.5
        128
    elseif density < 1.0
        256
    elseif density < 2.0
        512
    else
        1024
    end
    grid = n == 0 ? 0 : cld(n, block)
    return block, grid
end

macro sphota(n, density)
    return :(sphota_schedule($(esc(n)), $(esc(density))))
end

if ARGS[1] == "--schedule"
    length(ARGS) == 3 || error("schedule needs n and density")
    n = parse(Int, ARGS[2])
    density = parse(Float64, ARGS[3])
    block, grid = @sphota n density
    println(block, " ", grid)
    exit(0)
end

using LinearAlgebra

length(ARGS) >= 4 || error("Expected mode, input, output, N, lam")
mode, input, output = ARGS[1], ARGS[2], ARGS[3]
Base.ENDIAN_BOM == 0x04030201 || error("little-endian host required")
raw = read(input)
floats = collect(reinterpret(Float32, raw))

if mode == "damp"
    length(ARGS) == 5 || error("damp mode needs N and lam")
    n = parse(Int, ARGS[4])
    lam = parse(Float32, ARGS[5])
    length(floats) == 2n || error("damp input must be 2N floats")
    theta = floats[1:n]
    xi = floats[n+1:2n]
    damped = theta .* exp.(-lam .* xi)
    write(output, damped)
else
    error("unknown mode: $mode")
end
