function out = inv_haar_wavelet1D(f)

    n = size(f,2);          % input signal length
    h = [ 1 1]/sqrt(2);     % lowpass filter,  h_phi(n) in the book
    g = [-1 1]/sqrt(2);     % highpass filter, h_psi(-n) in the book  
    
    for j=1:log2(n)
        
        % select coarse part of the input
        Coarse = f(1:2^(j-1));
        
        % select detail part of the input
        Detail = f(2^(j-1)+1:2^j);
        
        % TODO 2
        
        % upsampling, TODO 2a
        up = zeros(1, 2^j -1);
        up(1:2:end) = Coarse;
        
        % lowpass filtering, TODO 2b
        lp = conv(up, h);

        % upsampling, TODO 2c
        up = zeros(1,2^j -1);
        up(1:2:end) = Detail;

        % highpass filtering, TODO 2d
        hp = conv(up, g);

        % summing up
        f(1:2^j) = lp(1:end) + hp(1:end);
        
    end
    out = f(end:-1:1);

end
