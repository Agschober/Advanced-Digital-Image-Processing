% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

% Step 0. Load and downsample the images
clear
close all

stack_path = '../focus_stack_adip/';
docs = dir(strcat(stack_path,'*.tif'));
nstack = size(docs,1);
filename = docs(1).("name");
im1 = imread(strcat(stack_path,filename));

[nx, ny, ncol] = size(im1);

clear im1;

scale = 10;
nx = round(nx/scale);
ny = round(ny/scale);

im_stack = struct; %zeros([nx,ny,ncol,nstack]);

tic
for i = 1:nstack
    im = imread(strcat(stack_path,docs(i).('name')));
    im = imfilter(im,fspecial('gaussian',7,1.),'same','replicate');
    im = imresize(im,[nx, ny],'bicubic');
    im_stack(i).image = im2double(im);
    im_stack(i).gs_image = color2grayscale(im_stack(i).image);
    %im_stack(:,:,:,i) = im2double(im);
    %gs_stack(:,:,i) = color2grayscale(im_stack(:,:,:,i));
end
toc

clear im;

%dipshow(gs_stack)

% Step 1. Compute the dense SIFT image

% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision

patchsize=8;
gridspacing=1;

num_angles = 8;
num_bins = 4;

%sift_stack = zeros([nx - (patchsize/2 +  2*gridspacing) ,ny - (patchsize/2 +  2*gridspacing),128,nstack]);
%normalized_sift_stack = sift_stack;

for i = 1:nstack
    [im_stack(i).sift, grid_x, grid_y] = dense_sift(im_stack(i).gs_image, patchsize, gridspacing);
    [nrows, ncols, cols] = size(im_stack(i).sift);
    sift_norm = reshape(im_stack(i).sift, [nrows*ncols num_angles*num_bins*num_bins]);
    sift_norm = normalize_sift(sift_norm);
    im_stack(i).normalized_sift = reshape(sift_norm, [nrows ncols num_angles*num_bins*num_bins]);
end

clear sift_norm;
%{
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
gs2 = warpImage(gs2, vx, vy);
Sift2 = warpImage(Sift2, vx, vy);
Sift2_norm = reshape(Sift2, [nrows*ncols num_angles*num_bins*num_bins]);
Sift2_norm = normalize_sift(Sift2_norm);
Sift2_norm = reshape(Sift2_norm, [nrows ncols num_angles*num_bins*num_bins]);

%The registerd image is just slightly smaller -> rescale 
gs1 = gs1(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);

[nx, ny] = size(gs1); 
%}

%Obtain activity maps
A = struct;

for i = 1:nstack
    sift = im_stack(i).sift;
    A(i).amap = sum(sift(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:), 3);
end

clear sift;

for i =1:nstack
    im_stack(i).gs_image = im_stack(i).gs_image(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:,:);
end

[nx, ny] = size(im_stack(1).gs_image);

%step 2: initial decision map
for i = 1:nstack
    A(i).M = zeros([nx - 2*(patchsize/2 +  2*gridspacing),ny - 2*(patchsize/2 +  2*gridspacing)]);
end


s = zeros(nstack,1);
for i = patchsize:nx - patchsize + 1
    for j = patchsize:ny - patchsize + 1
        for k = 1:nstack
            s(k) = sum(A(k).amap(i - patchsize + 1: i + 1, j - patchsize + 1: j + 1), [1,2]);
        end

        smax = squeeze(max(s));
        
        if size(smax) == 1
            index = find(s == smax);
            A(index).M(i - patchsize + 1: i + 1, j - patchsize + 1: j + 1) = A(index).M(i - patchsize + 1: i + 1, j - patchsize + 1: j + 1) + 1;
        end
    end
end
pause
%rough decision maps
for i = 1:nstack
    D(:,:,i) = sum(M, 3) - M(:,:,i) == 0;
end

dipshow(D)

%uncertainty mask
mask = sum(D,3) == 0;

%add post-processing step to D1 and D2
%remove small regions and holes in large areas (closing, opening or some other morphological operation)
n_erosion = 1;
dim_filt_x = ceil(nx/(10*n_erosion));
dim_filt_y = ceil(ny/(10*n_erosion));
filt_type = 'elliptic';

tic
for j =1:nstack
    D_tmp = D(:,:,j);
    for i = 1:n_erosion
        D_tmp = dilation(D_tmp,[dim_filt_x,dim_filt_y], filt_type);
    end
    for i = 1:n_erosion
        D_tmp = erosion(D_tmp, [dim_filt_x,dim_filt_y], filt_type);
    end
    for i = 1:n_erosion
        D_tmp = erosion(D_tmp, [dim_filt_x,dim_filt_y], filt_type);
    end
    for i = 1:n_erosion
        D_tmp = dilation(D_tmp,[dim_filt_x,dim_filt_y], filt_type);
    end
    D(:,:,j)
end
toc

clear D_tmp

%step 3: refine decision map using spatial frequency on image patch
cases = find(mask == 1);

i_indices = 1:nx - patchsize/2 - 2*gridspacing;
j_indices = 1:ny - patchsize/2 - 2*gridspacing;
%i_indices = 1:nx;
%j_indices = 1:ny;
[j_indices, i_indices] = meshgrid(j_indices,i_indices);
i_indices = i_indices(cases);
j_indices = j_indices(cases);

tic
for k = 1:size(cases,1)
    i = i_indices(k);
    j = j_indices(k);

    if i - patchsize/2 < 0
        i = patchsize/2;

    elseif i + patchsize/2 > ny
        i = nx - patchsize/2;

    end

    if j - patchsize/2 < 0
        j = patchsize/2;

    elseif j + patchsize/2 > nx
        j = ny - patchsize/2;

    end

    SF_stack = zeros(1,nstack);

    for l = 1:nstack
        gs = gs_stack(:,:,l);
        SF_stack(l) = SF(gs(i - patchsize/2 + 1 : i + patchsize/2, j - patchsize/2 + 1 : j + patchsize/2));
    
    SF_max = max(SF_stack);
    
    for m = 1:size(SF_max,1)
        index = find(SF_stack == SF_max(m));
        D(i,j,index) = 1/size(SF_max,1);
    end

    end
k/size(cases,1)
end
toc
clear gs; clear D_tmp

dipshow(D)

%step 4: fuse images
im_fin = sum(gs_stack(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:).*D,3);
figure; imshow(im_fin)


% Define functions used in script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grayscale = color2grayscale(I)

    %AS: taken from 'Detailed Fusion scheme' from assignment  (assumption is that channel numbers correspond to rgb)
    grayscale = 0.299*I(:,:,1) + 0.587*I(:,:,2) + 0.114*I(:,:,3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spatial_frequency = SF(I)
    
    %takes a grayscale 2D image and calculated the spatial frequency in the entire image.
    [N,M] = size(I);
    
    %definition of spatial frequency taken from assignment
    CF =  1/(N*M) * sum(sum( (I(2:end,:) - I(1:N-1,:)).^2 ));  %don't take the square root since it is squared in the next step anyways
    RF =  1/(N*M) * sum(sum( (I(:,2:end) - I(:,1:M-1)).^2 ));

    spatial_frequency = sqrt(CF + RF);
end



%% refine decision map
%
%% fuse images