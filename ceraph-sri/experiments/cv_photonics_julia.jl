# Independent CPU implementation; input and output are little-endian Float32 arrays.
using LinearAlgebra

length(ARGS) == 8 || error("Expected input, output, shots, Vs, Vn, eta, k, and r")
input, output = ARGS[1:2]
shots = parse(Int, ARGS[3])
vs, vn, eta, coupling, r = parse.(Float64, ARGS[4:8])
shots >= 2 || error("At least two shots are required")
Base.ENDIAN_BOM == 0x04030201 || error("This binary interchange requires a little-endian machine")
raw = read(input)
length(raw) == 6 * shots * sizeof(Float32) || error("Incorrect input size")
z = reshape(reinterpret(Float32, raw), shots, 6)
conditions = ((0.0, nothing), (0.0, 0.0), (r, nothing), (r, 0.0), (0.0, r))
result = Matrix{Float32}(undef, shots, length(conditions))

for (j, (ra, rb)) in enumerate(conditions)
    va = eta * (vs + vn + 0.5 * exp(-2ra)) + (1 - eta) / 2
    c = sqrt(eta) * vs
    if rb === nothing
        a, b = c / va, 0.0
    else
        vb = eta * (coupling^2 * vn + 0.5 * exp(-2rb)) + (1 - eta) / 2
        cross = eta * coupling * vn
        a, b = [va cross; cross vb] \ [c, 0.0]
    end
    for i in 1:shots
        s = sqrt(vs) * z[i, 1]
        n = sqrt(vn) * z[i, 2]
        ya = sqrt(eta) * (s + n + exp(-ra) * z[i, 3] / sqrt(2)) +
             sqrt((1 - eta) / 2) * z[i, 5]
        yb = rb === nothing ? 0.0 : sqrt(eta) * (coupling * n + exp(-rb) * z[i, 4] / sqrt(2)) +
             sqrt((1 - eta) / 2) * z[i, 6]
        result[i, j] = a * ya + b * yb
    end
end
write(output, vec(result))
