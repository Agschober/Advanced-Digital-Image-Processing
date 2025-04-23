% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

% Step 0. Load and downsample the images

im1=imread('flower-fg.jpg');
im2=imread('flower-bg.jpg');

[nx, ny, ncol] = size(im1);

% AS: for some reason the pictures from my phone weren't the same size, makes comparison difficult, implemented a small fix :)
scale = 3;
nx = round(nx/scale);
ny = round(ny/scale);

im1=imresize(imfilter(im1,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic');
im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic');

im1=im2double(im1);
im2=im2double(im2);

gs1 = color2grayscale(im1);
gs2 = color2grayscale(im2);

figure; imshow(im1);
figure; imshow(im2);

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

% step 1 +  IMAGE REGISTRATION
% AS: SIFT flow works REALLY WELL (as far as I know) FOR IMAGE REGISTRATION IT SEEMS -> not part of the assignment but definitely worth mentioning
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%AS: register SIFT images
Im2 = warpImage(im2, vx, vy);
Sift2 = warpImage(Sift2, vx, vy);
Sift2_norm = reshape(Sift2, [nrows*ncols num_angles*num_bins*num_bins]);
Sift2_norm = normalize_sift(Sift2_norm);
Sift2_norm = reshape(Sift2_norm, [nrows ncols num_angles*num_bins*num_bins]);

%Obtain activity maps
A1 = sum(Sift1, 3);
A2 = sum(Sift2, 3);

%step 2: initial decision map
M1 = zeros(size(A1));
M2 = M1;

for i = patchsize:nx - patchsize + 1
    for j = patchsize:ny - patchsize + 1
        s1 = sum(sum(A1(i - patchsize + 1: i, j - patchsize + 1: j)));
        s2 = sum(sum(A2(i - patchsize + 1: i, j - patchsize + 1: j)));
        
        if s1 > s2
            M1(i - patchsize + 1: i, j - patchsize + 1: j) = M1(i - patchsize + 1: i, j - patchsize + 1: j) + 1;
        elseif s2 > s1
            M2(i - patchsize + 1: i, j - patchsize + 1: j) = M2(i - patchsize + 1: i, j - patchsize + 1: j) + 1;
        end
    end
end

%rough decision maps
D1 = M2 == 0;
D2 = M1 == 0;

D_init = D1 + (1 - ((1-D2).*D1 + D2))*0.5;
figure;imshow(D_init)

%add pre-processing step to D1 and D2
%remove small regions and holes in large areas (closing, opening or some other morphological operation)
D1 = closing(D1, round(nx/100));
D2 = closing(D2, round(nx/100));

D_init = D1 + (1 - ((1-D2).*D1 + D2))*0.5;
D_init = im2mat(D_init);
figure;imshow(D_init)

%step 3: refine decision map
cases = find(D_init == 0.5);

for i = 1:size(cases)

end


%step 4: fuse images
im_fin = Im1.*D_init + (1-D_init).*warpI2;
figure; imshow(im_fin)