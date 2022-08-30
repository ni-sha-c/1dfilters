using CUDA
using Test
using BenchmarkTools

next(x, s) = (2*x + s*sin(16*x)/16) % 1
dnext(x, s) = abs(2 + s*cos(16*x))
d2next(x, s) = -16*s*sin(16*x)
sig_obs = 0.1
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

function get_emp_srb(nbins, ntime, nsamples, nrep, s)
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
function bin_fo_fi(fo_sa, fo_gr, fi_gr, y, nbins, nsamples, sig)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
    x = fo_sa[index]
	bin_no = Int(cld(x, 1/nbins)) 	
	fo_gr[(index - 1)*nbins + bin_no] += 1
	lkhd = exp(-(y-x)^2/(2*sig*sig))
	fi_gr[(index - 1)*nbins + bin_no] += lkhd
	return nothing
end
function post_process_fo_fi_cdfs(fo_gr, fi_gr, nbins)
	cdf = zeros(Float32, nbins)
	cdf_fo = zeros(Float32, nbins)

	for i = 1:nbins
		cdf[i] = sum(fi_gr[i:nbins:end])
		cdf_fo[i] = sum(fo_gr[i:nbins:end])
	end

	cdf = cumsum(cdf)/sum(cdf)
	cdf_fo = cumsum(cdf_fo)/sum(cdf_fo)
	return cdf, cdf_fo
end

function get_fo_fi_cdfs(fo_sa, y, nsamples, nbins)
	fo_gr = CUDA.fill(0.0f0, nbins*nsamples) 
	fi_gr = CUDA.fill(0.0f0, nbins*nsamples)
		
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	sig = sig_obs
	@show threads, blocks
	CUDA.@sync begin
			@cuda threads=threads blocks=blocks bin_fo_fi(fo_sa, fo_gr, fi_gr, y, nbins, nsamples, sig)
	end
	cdf, cdf_fo = post_process_fo_fi_cdfs(fo_gr, fi_gr, nbins)
end
function driver_cdfs(y, nsamples, nbins)
	fo_sa = CUDA.rand(nsamples)
    cdf, cdf_fo = get_fo_fi_cdfs(fo_sa, y, nsamples, nbins)
	return cdf, cdf_fo
end
function tran_lin_interp(fo_sa, fi_sa, cdf, cdf_fo, nbins)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	x = fo_sa[index]
	i2f = Int(cld(x, 1/nbins))
    x_mid = (i2f - 1)*(1/nbins) + 0.5/nbins
	Tx = x
	x1f, x2f = 0.0, 1.0
	c1f, c2f = 0.0, 1.0
	cf = 0.0
	
	if !(i2f == nbins)
	
		if x > x_mid 
		
			x1f = x_mid
			x2f = x_mid + (1/nbins)
		
			c1f = cdf_fo[i2f]
			c2f = cdf_fo[i2f + 1]
		
		else
			x1f = max(x_mid - (1/nbins), 0)
			x2f = x_mid
			c1f = (i2f > 1) ? cdf_fo[i2f-1] : 0.0
			c2f = cdf_fo[i2f]
		
		end
	
		cf = c1f + (c2f - c1f)*(x - x1f)/(x2f - x1f)
		i2 = 0
		for k = 1:nbins
			if (cdf[k] .>= cf)
				i2 = k
				break
			end
		end
		x2 = (i2 - 1)*(1/nbins) + 0.5/nbins
		x1 = max(x2 - (1/nbins), 0)  
    	c1 = (i2 > 1) ? cdf[i2 - 1] : 0.0
		c2 = cdf[i2]
		Tx = (cf - c1)*(x2 - x1)/(c2 - c1) + x1
		
	end
	fi_sa[index] = Tx
 	return nothing
end
function transport(y, nsamples, nbins)
	fo_sa = CUDA.rand(nsamples)
	fi_sa = CUDA.fill(0.0f0, nsamples)
	a_temp, b_temp = get_fo_fi_cdfs(fo_sa, y, nsamples, nbins)	
	cdf = CuArray(a_temp)
	cdf_fo = CuArray(b_temp)
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks tran_lin_interp(fo_sa, fi_sa, cdf, cdf_fo, nbins)
	end
	return fo_sa, fi_sa	
end
function evolve_dynamics(samples, ntime, s)
	index = threadIdx().x + (blockIdx().x - 1)*blockDim().x
	x = rand()
	for i = 1:ntime
		x = next(x,s)
	end
	samples[index] = x
	return nothing
end

function get_srb_samples(nsamples, ntime, s)
	samples = CUDA.fill(0.0f0, nsamples) 
	threads = min(nsamples, 1024)
	blocks = cld(nsamples, threads)
	CUDA.@sync begin
		@cuda threads=threads blocks=blocks evolve_dynamics(samples, ntime, s)
	end
	return samples
end

