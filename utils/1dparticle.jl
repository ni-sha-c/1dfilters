include("../examples/sawtooth.jl")
include("sir_1D.jl")
using PyPlot

function evolve_prob(s, N, T)
	x = rand(N)
	for t = 1:T
		for (i, xi) = enumerate(x)
			x[i] = next(xi, s)
		end
	end
	return x
end
function cdf(x, xarr)
	N = size(xarr)[1]
	return 1/N*sum(xarr .<= x)
end
function plot_dist(x, t, fig, ax)
	ax.hist(x, density=true, bins = 100, histtype="step", lw=3.0, 
			label=string(string(t)))
end

