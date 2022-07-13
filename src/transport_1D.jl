include("../examples/sawtooth.jl")
using JLD
σ_o = 0.05
function generate_data(T, s)
	x = rand(T)
	y = rand(T)
	y[1] = (x[1] + σ_o*randn())%1
	for k = 1:(T-1)
		x[k+1] = next(x[k], s)
		y[k+1] = (x[k+1] + σ_o*randn())%1
	end
	return x, y
end
function log_likelihood(a)
		return log(1/sqrt(2π)/σ_o) -0.5*a*a/σ_o/σ_o
end
function cdf(x, xarr)
	N = size(xarr)[1]
	return 1/N*sum(xarr .<= x)
end
function posterior(xarr, y)
	N = size(xarr)[1]
	a_n = y .- xarr
	lklhd = log_likelihood.(a_n)
	prob = (1/N)*exp.(lklhd)
	prob ./= sum(prob)
	return prob
end
function linear_approx_of_cmf(x1, x2, c1, c2, c)
	return (c - c1)/(c2 - c1)*(x2 - x1) + x1
end
function get_tran(cmf, x, pos, cmff)
	cmf_temp = copy(cmf)
	x_temp = copy(x)
	insert!(cmf_temp, 1, 0)
	push!(cmf_temp, 1.0)
	insert!(x_temp, 1, 0)
	push!(x_temp, 1)
	@show cmf_temp, x_temp
	x1, x2 = x_temp[pos], x_temp[pos+1]

	c1, c2 = cmf_temp[pos], cmf_temp[pos+1]
	return linear_approx_of_cmf(x1, x2, c1, c2, cmff)
end
function opt_tran(c_pr, c_po, x)
	Tx = similar(x)
	f_c = 0
	pos = 1
	for (k, cmff) in enumerate(c_pr)
		for (l, cmf) in enumerate(c_po)
			if cmf >= cmff
				f_c = cmff
				pos = l
				break
			end
		end
		Tx[k] = get_tran(c_po, x, pos, f_c)
	end
	return Tx
end
function transport_map(x_pr, post_pr)
	N = size(x_pr)[1]
	pr_pr = (1/N)*ones(N)
	cmf_pr = cumsum(pr_pr)
	cmf_post = cumsum(post_pr)
	return opt_tran(cmf_pr, cmf_post, x_pr)
end
function analysis(y, x)
	
end
function transport_analysis(y, x)
	post_pr = posterior(x, y)
	return transport_map(x, post_pr)
end


function forecast(x, s, τ)
	for t = 1:τ
		x .= next.(x, s)
	end
	return x
end
function transport_filter(x,y,τ,T,N,s)
	x_pr = zeros(N,T)
	for t = 1:T
		x .= forecast(x, s, τ)
		x .= transport_analysis(y[t], x)
		x_pr[:,t] .= x
    end
	return x_pr
end
function assimilate(N, Nthr, τ, T, s)
	x_t, y = generate_data(T, s)
	x = rand(N)
	x .= forecast(x, s, 1000)
	x_pr = transport_filter(x,y,τ,T,N,s)
	return x_pr, x_t, y 
end
