function out = haar_wavelet1D(f)

n = size(f,2);       % input signal length
out=[];              % output variable
h = [ 1 1]/sqrt(2);  % lowpass filter,  h_phi(n) in the book 7-146
g = [-1 1]/sqrt(2);  % highpass filter, h_psi(-n) in the book 7-147

for i=1:log2(n)
    % TODO 1
    
    % lowpass filtering, TODO 1a (note that conv(u,v) outputs a vector of length lengtt(u) + length(v) - 1)
    % since in this case the filter is length 2, this means the filtered output has 1 more element than the input.
    lp = conv(h, f);
    
    % downsampling, TODO 1b
    dslp = lp(2:2:end);
    
    % highpass filtering, TODO 1c
    hp = conv(g, f);
    
    % downsampling, TODO 1d
    dshp = hp(2:2:end);
    
    % output at the current scale, TODO 1e
    
    out = [dshp out]; %I don't understand what the scaling and detail coefficients are supposed to represent (are they just samples, or are they inner products like in a Fourier transform?)
    
    %compare to haart(f) from wavelet toolbox
    %[a, d] = haart(f)
    % a = approximation coefficient at last level (in this case a single number) (in this case this is last low pass filtered approximation.)
    % for the example the d should be 3 elements -> [4 -3/sqrt(2) -3/sqrt(2)]
    % d = detail coefficient, has shape {[2^N], [2^{N-1}, ..., [1]}, each vector is the list of detail coeffs at a certain scale.
    % but for 1D this should always be 2?

    %
    Coarse = [dslp];
    % replace f with the coarse downsampled scale and iterate the
    % procedure
    f = Coarse;
    
end
out = [f out]; %Is this even necessary? I thought this should just work if you have 'out'