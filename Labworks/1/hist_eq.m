function out = hist_eq(in)

    [N, M] = size(in);
    out = zeros(N, M);

    % TODO3
    % compute the image histogram
    hist  = zeros(1, 256);        
    for i=1:M
        for j=1:N
            grayValue = in(j,i); %range is 0-255
            hist(grayValue+1) = hist(grayValue+1) + 1; %TODO3a
        end
    end

    % perform histogram equalization
    for i = 1:M
        for j = 1:N
            sum = 0;
            grayValue = in(j,i);
            % here the CDF is computed
            for k = 1:grayValue
                sum = sum + hist(k);%TODO3b
                %TODO3c
            end
            out(j,i) = sum/N/M;
        end
    end

    % rescale the intensities to [0, 255]
    out = out*255; %TODO3d
    
end
