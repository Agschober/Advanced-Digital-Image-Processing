% Programming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: S. Haghparast
% Year:2023
%
% Labwork #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1              Fourier transform interpretation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
im1 = readim('FFT_sample1.ics');
im2 = readim('FFT_sample2.ics');

dipshow(im1)
title('FT image 1')
dipshow(im2)
title('FT image 2')

%Think how to set the display range. The Fourier transformation of an image
%is complex. How to display the image??

% Solution: find the frequency as the peak position - center of the image (128),

% TODO 1
% Calculate the spatial frequency of the peaks
center = [128,128];
m1 = [-8,0],[8,0];%
m2 = [0,-32],[0,32];%

% TODO 2
% What is the maximum spatial frequency in an image of 256x256 pixels?

N=256;
m=[128]; % TODO 2
out1 = cos(2*pi*m/N*(xx([N,N],'true')));
dipshow(out1)
dipshow(ft(out1))


% TODO 3
% Compute images with the estimated frequencies of TODO 1 and compare them
% with the IFT of the input


dipshow(ift(im1, 'real'))
title('IFT image 1')
dipshow(real(ift(im2)))
title('IFT image 2')

est_im1 = cos(2*pi*m1(1)/N*xx([N,N],'true'));
est_im2 = sin(2*pi*m2(4)/N*yy([N,N],'true'));

dipshow(est_im1)
title('est IFT image 1')
dipshow(est_im2)
title('est IFT image 2')


%TODO 4
a = readim('pout.tif');
b = ft(a);

parseval_a = sum(abs(im2mat(a)).^2, "all") % int(i)dx dy
parseval_b = sum(abs(im2mat(b)).^2, "all")/(prod(size(a))) % int(I)df_x df_y

pause;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2  Demo            Fourier transform and filtering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;

% Read the input image
im_color = readim('Zebra.jpg') % the input image clearly shows jpg compression artifacts in the back (false colors)
im = (im_color{1}+im_color{2}+im_color{2})/3 

% Compute fourier transform of the image using dipimage
Fim = ft(im); 
dipshow(Fim)

% Creating hard-low pass filter in frequency domain 
% TODO 5
value = 64;

N = size(im);
hard_low = rr(imsize(im))<value;
dipshow(hard_low)
title('Hard Lowpass filter');

out1=ift(hard_low*Fim);
dipshow(out1) 
title('Hard Low'); 


%  Creating soft-low pass filter in frequency domain 
N = size(im);
soft_low= exp(-rr(imsize(im)).^2/(2*value^2));
dipshow(soft_low)
title('Soft Lowpass filter');

out2 = ift(soft_low*Fim);
dipshow(out2) 
title('Soft Low') ;

%  Creating soft-high pass filter in frequency domain 
% TODO 6
soft_high = 1 - soft_low;
dipshow(soft_high)
title('Soft Highpass filter');

out3 = ift(soft_high*Fim);
dipshow(out3)
title('soft high')

out4 = out3 + out2;
title('low and high')
dipshow(out4);

dipshow(im);

%check that low and high sum up to the original

