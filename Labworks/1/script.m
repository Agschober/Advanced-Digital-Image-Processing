% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2021
%
% Assignment #1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the input sythetic image.

I1 = readim('chirp.tiff');
dipshow(I1);
%%

% define scale factor
scale = 3;

% initial image dimensions
[col1, row1] = imsize(I1);

% new image dimensions
col2 = scale * col1;
row2 = scale * row1;

%%
% (a) nearest neighbour interpoaltion

% initialize upsampled image
I2 = newim(col2,row2);

% nested loops are slow in dipimage, we do this in matlab using im2mat()
% function. Type 'help im2mat()' for more information'
I1mat = im2mat(I1);
I2mat = im2mat(I2);

for j=1:col2
    for i=1:row2
       
        % TODO1
        u = ceil(i/scale);
        v = ceil(j/scale);
        
        I2mat(i,j) = I1mat(u,v);
    end
end

% now convert matrix I2mat to dipimage I2
I2 = mat2im(I2mat);

% plot interpolated image
dipshow(I2)
%%
% (b) bilinear interpolation

% initialize upsampled image
I3 = newim(col2,row2);

% nested loop are slow in dipimage, we do this in matlab using im2mat()
% function. Type 'help im2mat()' for more information'
I1mat = im2mat(I1);
I3mat = im2mat(I3);

for j = 1:col2
	for i = 1:row2
        % compute the coordinates of each point in original image 
		x = j/scale;
		y = i/scale;
        
		% find the nearest integer coordinates
		x0 = floor(x);
		y0 = floor(y);
        
		% check for the boundaries. 
        % (y0,x0) should be larger than (0,0) and smaller than (N1,M1)
		if x0 < 1; x0 = 1; end
        if x0 > col1-1; x0 = col1-1; end
		if y0 < 1; y0 = 1; end
		if y0 > row1-1; y0 = row1-1; end
        
		% compute the weights (s, t) according to the lecture note
		s = x - x0;
		t = y - y0;
        
		% implement bilinear interpolation
        if x0+2 > col1 || y0+2 > row1
            break
        end
        NW = I1mat(1+x0,2+y0);
        NE = I1mat(2+x0,2+y0);
        SW = I1mat(1+x0,1+y0);
        SE = I1mat(2+x0,1+y0);

        S = SW*(1-s) + SE*s;
        N = NW*(1-s) + NE*s;

        I3mat(i,j) = S*(1-t) + N*t;    
	end
end

% now convert matrix I3mat to dipimage I3
I3 = mat2im(I3mat);

% plot interpolated image
dipshow(I3)

pause % pauses the script and waits for key stroke
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read the input image; try different MATLAB built-in images
I1im = readim('pout.tif');
I1 = im2mat(I1im); % again convert as we need to loop in hist_eq

% apply histogram equalitzation
I2 = hist_eq(I1);

% plot the result and compare it with MATLAB built-in function
h1 = dipshow(I1im);
h1.Name = 'original image';

h2 = dipshow(mat2im(I2));
h2.Name = 'Histogram equalized image';
%%
% use dipimage built-in function for comparison
I3 = hist_equalize(I1);
h3 = dipshow(I3);
h3.Name = "dipimage histeq()";

% add image histograms
diphist(mat2im(I1))
title('Original image histogram')
xlabel('intensity');ylabel('count')

diphist(mat2im(I2))
title('Image histogram after equalization');
xlabel('intensity');ylabel('count')

diphist(I3)
title('Image histogram after equalization (dipimage)')
xlabel('intensity');ylabel('count')
