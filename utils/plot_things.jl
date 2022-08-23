using PyPlot
using JLD
function plot_emp_den()
		X = load("../data/rho_0.1_67bil.jld")
		rho = X["rho"]
		nbins = length(rho)
		x = LinRange(0, 1.0, nbins)
		fig, ax = subplots()
		ax.plot(x, rho, ".", ms=20.0)
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel("Empirical density at s = 0.1", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
end



