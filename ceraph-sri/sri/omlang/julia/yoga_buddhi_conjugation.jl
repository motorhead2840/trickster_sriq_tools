# Sphota Julia conjugation kernel — latent Ĉ_S / Υ_YB and observable Υ_obs.
# Binary interchange: little-endian Float32 blocks (matches cv_photonics_julia.jl style).
#
# Args: mode input output [extra...]
#   mode=oracle: input = [v..., xi...] (2N floats), output = [thesis..., antithesis..., samya...] (3N)
#   mode=obs:    input = [ya..., yb..., w0, w1] (2N+2), output = [antithesis..., projected..., mmse...] (3N)

using LinearAlgebra

length(ARGS) >= 3 || error("Expected mode, input, output [, N]")
mode, input, output = ARGS[1:3]
Base.ENDIAN_BOM == 0x04030201 || error("little-endian host required")
raw = read(input)
floats = collect(reinterpret(Float32, raw))

if mode == "oracle"
    length(ARGS) == 4 || error("oracle mode needs N")
    n = parse(Int, ARGS[4])
    length(floats) == 2n || error("oracle input must be 2N floats")
    v = floats[1:n]
    xi = floats[n+1:2n]
    thesis = v .+ xi
    antithesis = v .- xi
    samya = 0.5f0 .* (thesis .+ antithesis)
    write(output, vcat(thesis, antithesis, samya))
elseif mode == "obs"
    length(ARGS) == 4 || error("obs mode needs N")
    n = parse(Int, ARGS[4])
    length(floats) == 2n + 2 || error("obs input must be 2N+2 floats")
    ya = floats[1:n]
    yb = floats[n+1:2n]
    w0, w1 = floats[2n+1], floats[2n+2]
    antithesis = (2f0 * w0 - 1f0) .* ya .+ (2f0 * w1) .* yb
    projected = 0.5f0 .* (ya .+ antithesis)
    mmse = w0 .* ya .+ w1 .* yb
    write(output, vcat(antithesis, projected, mmse))
else
    error("unknown mode: $mode")
end
