include("../examples/sawtooth.jl")
using JLD
σ_o = 0.05

function generate_data(T, s, τ=1)
	x = rand(T)
	y = rand(T)
	y[1] = (x[1] + σ_o*randn())%1
	for k = 1:(T-1)
		x0 = x[k]
		x[k+1] = forecast([x0], s, τ)[1]
		y[k+1] = (x[k+1] + σ_o*randn())%1
	end
	return x, y
end
function log_likelihood(a)
		return log(1/sqrt(2π)/σ_o) -0.5*a*a/σ_o/σ_o
end
function forecast(x0, s, τ)
	x = copy(x0)
	for t = 1:τ
		x .= next.(x, s)
	end
	return x
end
function assimilate(N, Nthr, τ, T, s)
	x_t, y = generate_data(T, s)
	x = rand(N)
	x .= forecast(x, s, 1000)
	x_pr = transport_filter(x,y,τ,T,N,s)
	return x_pr, x_t, y 
end

