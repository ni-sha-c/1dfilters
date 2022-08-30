using PyPlot
using JLD
function plot_emp_den()
		X = load("../data/rho_0.1_67bil.jld")

		rho = X["rho"]
		nbins = length(rho)
		x = LinRange(0, 1.0, nbins)
		fig, ax = subplots()
		ax.plot(x, rho, ".", ms=20.0, label="s = 0.1")
		

		X = load("../data/rho_0_67bil.jld")
		rho = X["rho"]
		nbins = length(rho)
		x = LinRange(0, 1.0, nbins)
		ax.plot(x, rho, ".", ms=20.0, label="s = 0")
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel("Empirical density", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		ax.legend(fontsize=30)
end

function plot_den_grad()
		X = load("../data/g_0.6.jld")

		g = X["g"]
		nbins = length(g)
		x = LinRange(0, 1.0, nbins)
		fig, ax = subplots()
		ax.plot(x, g, ".", ms=20.0, label="finite difference")
		
		X = load("../data/g_0.6_formula.jld")

		g = X["g"]
		nbins = length(g)
		x = LinRange(0, 1.0, nbins)
		#x = X["x"]
		ax.plot(x, g, ".", ms=20.0, label="formula")
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel("Score", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		ax.legend(fontsize=30)


end

function plot_cdfs()
		X = load("../data/cdfs.jld")
		cdf = X["cdf"]
		cdf_fo = X["cdf_fo"]

		fig, ax = subplots()

		nbins = length(cdf)
		x = LinRange(0, 1.0, nbins)
		ax.plot(x, cdf, ".", ms=20.0, label="filtering")
		ax.plot(x, cdf_fo, ".", ms=20.0, label="forecast")
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel("CDFs", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		ax.legend(fontsize=30)
end

function plot_transport()
		X = load("../data/fo_sa_fi_sa.jld")
		x = X["x"]
		Tx = X["Tx"]

		fig, ax = subplots()

		ax.plot(x[1:10:end], Tx[1:10:end], ".", ms=2.0)
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel("T(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		#ax.legend(fontsize=30)
end
