using CUDA
using Test
using BenchmarkTools

next(x, s) = (2*x + s*sin(16*x)/16) % 1
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
	logr = log.(r)
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
