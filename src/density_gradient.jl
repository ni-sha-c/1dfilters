using CUDA
using Test
using BenchmarkTools

next(x, s) = (2*x + s*sin(16*x)/16) % 1
dnext(x, s) = abs(2 + s*cos(16*x))
d2next(x, s) = -16*s*sin(16*x)
function hist_single_orbit(rho, ntime, nbins, s)
    x = rand()
	index = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
    for t = 1:500
		x = next(x,s)
	end
	for t = 1:ntime
		x = next(x,s)
		bin_no = Int(cld(x, 1.0/nbins))
		rho[(index - 1)*nbins + bin_no] += 1.0
	end    
	return nothing	
end 
function post_process_rho(rho, nbins, ntime, nsamples, nrep)
	r = zeros(nbins)
	for i = 1:nbins
		r[i] = sum(rho[i:nbins:end])/(nsamples*ntime*nrep)*nbins
	end
	return r	
end
function post_process_g(rho, nbins, ntime, nsamples, nrep)
	dlogr = zeros(nbins)
    r = post_process_rho(rho, nbins, ntime, nsamples, nrep)
	#logr = log.(r)
	logr = r
	for i = 2:nbins-1
		dlogr[i] = (logr[i+1] - logr[i-1])/2*nbins
	end
    dlogr[1] =  (logr[2] - logr[1])*nbins
    dlogr[nbins] =  (logr[nbins] - logr[nbins-1])*nbins
	return dlogr	
end

function get_emp_den(nbins, ntime, nsamples, nrep, s)
	nrho = nsamples * nbins
	rho = CUDA.fill(0.0f0, nrho)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)

	@show threads, blocks
	for n = 1:nrep
		CUDA.@sync begin
			@cuda threads=threads blocks=blocks hist_single_orbit(rho, ntime, nbins, s)
		end
	end	
	rho_final = post_process_rho(rho, nbins, ntime, nsamples, nrep)
	return rho_final
end
function get_den_grad(nbins, ntime, nsamples, nrep, s)
	nrho = nsamples * nbins
	rho = CUDA.fill(0.0f0, nrho)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)

	@show threads, blocks
	for n = 1:nrep
		CUDA.@sync begin
			@cuda threads=threads blocks=blocks hist_single_orbit(rho, ntime, nbins, s)
		end
	end	
	g = post_process_g(rho, nbins, ntime, nsamples, nrep)
	return g
end
function next_g(g, x, s)
	# takes g(x) and returns g(next(x,s))
	a = dnext(x, s) 
	c = d2next(x, s) 
	g = g/a - c/a/a
	return g
end
function post_process_g_cpu(x, g, nbins)
    g_gr = zeros(nbins)
    ntime = length(g)
	for t = 1:ntime
		bin_t = Int(cld(x[t], 1/nbins))
		g_gr[bin_t] += g[t]*nbins/ntime
	end
	return g_gr
end
function get_den_grad_cpu(norbit, nbins, s)
	x = rand()
	g = 0.0
	for t = 1:500
		g = next_g(g, x, s)
		x = next(x, s)
	end
	orbit = zeros(norbit)
	g_orbit = zeros(norbit)
	
	for t = 1:norbit
		g_orbit[t] = g
		orbit[t] = x
		g = next_g(g, x, s)
		x = next(x, s)
	end
	g_gr = post_process_g_cpu(orbit, g_orbit, nbins)
	return g_gr
end
