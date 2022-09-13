using PyPlot
using JLD
next(x,s) = (2*x + s*sin(16*x)/16 + 2) % 1
function plot_emp_den()
		X = load("../data/rho_tent_0.1.jld")

		rho = X["rho"]
		nbins = length(rho)
		x = LinRange(0, 2.0, nbins)
		fig, ax = subplots()
		ax.plot(x, rho/2.0, ".", ms=20.0, label="s = 0.1")
		
		
		X = load("../data/rho_tent_0.05.jld")
		rho = X["rho"]
		nbins = length(rho)
		x = LinRange(0, 2.0, nbins)
		ax.plot(x, rho/2.0, ".", ms=20.0, label="s = 0.05")
		
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
		X = load("../data/transport.jld")
		x = X["x"]
		Tx = X["Sx"]

		fig, ax = subplots()

		ax.plot(x[1:10:end], Tx[1:10:end], ".", ms=2.0)
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel(L"T^{-1}(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		#ax.legend(fontsize=30)
end

function plot_srb()
		X = load("../data/srb_samples_0.8.jld")
		x = X["x"]
		

		fig, ax = subplots()

		ax.hist(x,bins=200,density=true)
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel(L"\mu(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		#ax.legend(fontsize=30)
end
function plot_transport_with_slopes()
		X = load("../data/transport.jld")
		x = X["x"]
		Tx = X["Sx"]
		dx = X["dx"]
		fig, ax = subplots()

		ax.plot(x[1:100:end], Tx[1:100:end], ".", ms=5.0)
		epsi = 3.e-4
		Txp = Tx[1:1000:end] .+ epsi.*dx[1:1000:end]
		Txm = Tx[1:1000:end] .- epsi.*dx[1:1000:end]
		xp = x[1:1000:end] .+ epsi
		xm = x[1:1000:end] .- epsi
		ax.plot([xm, xp], [Txm, Txp], "k-", alpha=0.5) 
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel(L"T^{-1}(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		#ax.legend(fontsize=30)
end
function transport_ansatz()
		X = load("../data/transport.jld")
		x = X["x"]
		Tx = X["Sx"]
		rho_x = X["rho_x"]
		rho_Sx = X["rho_Sx"]

		fig, ax = subplots()
		nskip = 100
		x = x[1:nskip:end]
		Tx = Tx[1:nskip:end]
		ax.plot(Tx, x, ".", ms=5.0, label="opt T")
		ax.plot(Tx, Tx, "o", ms=5.0, label="identity")
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel(L"T\circ \varphi(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		ax.legend(fontsize=30)
	
		a1, b1 = minimum(Tx), maximum(Tx)
		a2, b2 = minimum(x), maximum(x)

		nbins = size(rho_x)[1]
		x1_gr = LinRange(a1, b1, nbins)
		x2_gr = LinRange(a2, b2, nbins)


		fig, ax = subplots()
		ax.plot(x1_gr, rho_Sx, ".", ms=5.0, label=L"\pi_{T-1}", color="blue")
		ax.fill_between(x1_gr, 0, rho_Sx, color="blue")
		
		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel(L"\rho(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		ax.legend(fontsize=30)

		fig, ax = subplots()
		
		ax.plot(x2_gr, rho_x, ".", ms=5.0, label=L"\pi_T", color="red")
		ax.fill_between(x2_gr, 0, rho_x, color="red")

		ax.set_xlabel("x", fontsize=30)
		ax.set_ylabel(L"\rho(x)", fontsize=30)
		ax.xaxis.set_tick_params(labelsize=30)
		ax.yaxis.set_tick_params(labelsize=30)
		ax.grid(true)
		ax.legend(fontsize=30)

end

