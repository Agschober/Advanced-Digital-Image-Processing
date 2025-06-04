% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

%%
close all
stack_path = '../focus_stack_adip/';
docs = dir(strcat(stack_path,'*.tif'));

%DEBUGGING
if isempty(docs)
    error('No .tif files found in the specified path.');
end

nstack = size(docs,1);
filename = docs(1).("name");
im1 = imread(strcat(stack_path,filename));
[nx_orig, ny_orig, ncol_orig] = size(im1);
clear im1;

%scaling as theyre really large
scale = 10;
nx = round(nx_orig/scale); 
ny = round(ny_orig/scale);

im_stack = struct([]);
color_stack_for_display = zeros(nx, ny, 3, nstack, 'like', im2double(imread(strcat(stack_path, docs(1).('name')))));
gs_stack =  zeros(nx, ny, nstack, 'like', im2double(imread(strcat(stack_path, docs(1).('name')))));


tic
for i = 1:nstack
    current_filename = docs(i).('name');
    im = imread(strcat(stack_path, current_filename));
    im = imfilter(im, fspecial('gaussian', 7, 1.0), 'same', 'replicate');
    im = imresize(im, [nx, ny], 'bicubic');

    im_stack(i).image = im2double(im); % Store the filtered, resized, color image
    im_stack(i).gs_image = color2grayscale(im_stack(i).image); %graysclae
    
    gs_stack(:,:,i) = im_stack(i).gs_image; % Fill gs_stack here
    color_stack_for_display(:,:,:,i) = im_stack(i).image;
    
    im_stack(i).filename = current_filename;
    im_stack(i).original_size = [nx_orig, ny_orig];
end
toc
clear im current_filename;

%display stack
dipshow(color_stack_for_display);
%single slice
%dipshow(im_stack(1).image);



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
%%
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

%% Step 2: Initial decision map
decision_radius = patchsize;
[nx_crop, ny_crop] = size(A(1).amap);  

for i = 1:nstack
    A(i).M = zeros(nx_crop - 2 * decision_radius, ny_crop - 2 * decision_radius);
end

s = zeros(nstack, 1);
for i = decision_radius + 1 : nx_crop - decision_radius
    for j = decision_radius + 1 : ny_crop - decision_radius
        for k = 1:nstack
            patch = A(k).amap(i - decision_radius : i + decision_radius, ...
                              j - decision_radius : j + decision_radius);
            s(k) = sum(patch, 'all');
        end

        [~, index] = max(s);
        A(index).M(i - decision_radius, j - decision_radius) = ...
            A(index).M(i - decision_radius, j - decision_radius) + 1;
    end
end

%figure; imagesc(A(2).M); axis image; colormap jet; title('Decision Map for Slice 1');
decision_stack = zeros(size(A(1).M,1), size(A(1).M,2), nstack);

for i = 1:nstack
    decision_stack(:,:,i) = A(i).M;
end

%dipshow(decision_stack);

%% Rough decision maps and uncertainty mask + processing

% Step 1: Extract all decision maps from struct A into a 3D matrix M
M = zeros(size(A(1).M,1), size(A(1).M,2), nstack);
for i = 1:nstack
    M(:,:,i) = A(i).M;
end

% Step 2: Rough decision maps D
D = false(size(M)); % initialize logical array
for i = 1:nstack
    D(:,:,i) = (sum(M, 3) - M(:,:,i)) == 0;
end

% Step 3: Display decision maps
dipshow(D);

% Step 4: Uncertainty mask
mask = sum(D, 3) == 0;

% Step 5: Morphological post-processing on D
n_erosion = 1;
dim_filt_x = ceil(nx / (10 * n_erosion));
dim_filt_y = ceil(ny / (10 * n_erosion));
filt_type = 'elliptic';

tic
for j = 1:nstack
    D_tmp = D(:,:,j);
    D(:,:,j) = medif(D_tmp, [dim_filt_x, dim_filt_y], filt_type);
end
toc

clear D_tmp

%Step 3: refine decision map using spatial frequency on image patch
cases = find(mask == 1);

i_indices = 1:nx - patchsize/2 - 2*gridspacing;
j_indices = 1:ny - patchsize/2 - 2*gridspacing;
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
    end

    SF_max = max(SF_stack);
    index = find(SF_stack == SF_max);
    D(i,j,index) = 1 / numel(index); % Normalize 


%     SF_max = max(SF_stack);
%     for m = 1:size(SF_max,1)
%         index = find(SF_stack == SF_max(m));
%         D(i,j,index) = 1/size(SF_max,1);
%     end
% 
%     end
% k/size(cases,1)
% end
  
end
toc

clear gs; clear D_tmp
%%%%%%%%%%%%%%%%%%%%foutinitus
dipshow(D)
%%
% Step 4: Fuse images
% Compute the cropping margins
crop_x = floor((size(gs_stack,1) - size(D,1)) / 2);
crop_y = floor((size(gs_stack,2) - size(D,2)) / 2);

% Crop gs_stack to match D's size
gs_cropped = gs_stack(crop_x+1:end-crop_x, crop_y+1:end-crop_y, :);

% Now fuse the grayscale images using D
im_fin = sum(gs_cropped .* D, 3);

% Display the fused image
figure; imshow(im_fin, []);



%im_fin = sum(gs_stack(patchsize+1:end - patchsize, patchsize+1:end- patchsize, :) .* D, 3);
%figure; imshow(im_fin)


% Define functions used in script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grayscale = color2grayscale(I)
    % AS: taken from 'Detailed Fusion scheme' from assignment
    grayscale = 0.299*I(:,:,1) + 0.587*I(:,:,2) + 0.114*I(:,:,3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spatial_frequency = SF(I)
    % Takes a grayscale 2D image and calculates the spatial frequency
    [N,M] = size(I);
    CF =  1/(N*M) * sum(sum((I(2:end,:) - I(1:N-1,:)).^2));
    RF =  1/(N*M) * sum(sum((I(:,2:end) - I(:,1:M-1)).^2));
    spatial_frequency = sqrt(CF + RF);
end

