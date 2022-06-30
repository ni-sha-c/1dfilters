include("../examples/sawtooth.jl")
include("../src/sir_1D.jl")

using PyPlot
function cdf(x, xarr)
	N = size(xarr)[1]
	return 1/N*sum(xarr .<= x)
end
function plot_dist(x, t, ax)
	ax.hist(x, density=true, bins = 100, histtype="step", lw=3.0, 
			label=string("time ", string(t)))
end
function plot_part(x, w, t, ax)
	ax.hist(x, weights=w, density=true, bins = 100, histtype="step", lw=3.0, 
			label=string("time ", string(t)))
end

function plot_filtering_recursion(T)
	x_pr, w_pr, x_t, y = assimilate(90000, 100, 1, T, 0.2)
	for t = 1:T
		fig, ax = subplots(1,2)
		plot_part(x_pr[:,t], w_pr[:,t], t, ax[1])
		ax[1].plot(x_t[t], 5, ".", ms=15,label="true")
		ax[1].plot(y[t], 5, ".", ms=15,label="observed")
		ax[2].hist(w_pr[:,t], density=true)
		ax[1].legend(fontsize=16)
	end

	return x_pr, w_pr, x_t, y
end
