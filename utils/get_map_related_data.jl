include("../src/transport_1D_gpu.jl")
using JLD
function save_map(nsamples, a, b, s)
	x = a .+ (b-a).*CUDA.rand(nsamples)
	y = Array(x)
	forecast(x, nsamples, 1, s)
	x = Array(x)
	println("Saving ", nsamples, " samples of x and phi(x)")
	save("../data/map.jld", "s", s, "name", "Pinched tent map", "Fx", x, "x", y)
end   
