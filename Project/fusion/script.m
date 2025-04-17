% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

% Step 0. Load and downsample the images

im1=imread('street-bg.jpg');
im2=imread('street-fg.jpg');

[nx, ny, ncol] = size(im1);

% AS: for some reason the pictures from my phone weren't the same size, make comparison difficult, implemented a small fix :)
scale = 3;
nx = round(nx/scale);
ny = round(ny/scale);

im1=imresize(imfilter(im1,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic');
im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic');

im1=im2double(im1);
im2=im2double(im2);

gs1 = color2grayscale(im1);
gs2 = color2grayscale(im2);

figure; imshow(gs1);
figure; imshow(gs2);

% Step 1. Compute the dense SIFT image

% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision

patchsize=8;
gridspacing=1;

Sift1=dense_sift(im1,patchsize,gridspacing);
Sift2=dense_sift(im2,patchsize,gridspacing);

%AS: normalization removed from dense_sift function, So Sift is full scale vector Sift_norm is normalized vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_angles = 8;
num_bins = 4;
[nrows, ncols, cols] = size(Sift1);

% normalize SIFT descriptors
Sift1_norm = reshape(Sift1, [nrows*ncols num_angles*num_bins*num_bins]);
Sift1_norm= normalize_sift(Sift1_norm);
Sift1_norm = reshape(Sift1_norm, [nrows ncols num_angles*num_bins*num_bins]);

[nrows, ncols, cols] = size(Sift2);

% normalize SIFT descriptors
Sift2_norm = reshape(Sift2, [nrows*ncols num_angles*num_bins*num_bins]);
Sift2_norm = normalize_sift(Sift2_norm);
Sift2_norm = reshape(Sift2_norm, [nrows ncols num_angles*num_bins*num_bins]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% visualize the SIFT image
figure;imshow(showColorSIFT(Sift1_norm));title('SIFT image 1');
figure;imshow(showColorSIFT(Sift2_norm));title('SIFT image 2');

% AS: SIFT flow works REALLY FUCKING WELL FOR IMAGE REGISTRATION IT SEEMS -> not part of the assignment but definitely worth mentioning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3. SIFT flow matching

% prepare the parameters
SIFTflowpara.alpha=2;
SIFTflowpara.d=40;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=60;

tic;[vx,vy,energylist]=SIFTflowc2f(Sift1_norm,Sift2_norm,SIFTflowpara);toc

% Step 4.  Visualize the matching results
Im1=im1(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);
Im2=im2(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);
warpI2=warpImage(Im2,vx,vy);
figure;imshow(Im1);title('Image 1');
figure;imshow(warpI2);title('Warped image 2');

% display flow
clear flow;
flow(:,:,1)=vx;
flow(:,:,2)=vy;
figure;imshow(flowToColor(flow));title('SIFT flow field');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;imshow(color2grayscale(Im1));title('Image 1');
figure;imshow(color2grayscale(warpI2));title('Warped image 2');