%A script to demonstrate the steps made to construct a SIFT image
clear
close all

I = imread('../standard_issue/erika.tif');
patch_size = 8;
grid_spacing = 1;


I = double(I);
I = mean(I,3);
I = I /max(I(:)); %image is normalized

dipshow(I);
% parameters
num_angles = 8;
num_bins = 4;
num_samples = num_bins * num_bins;
alpha = 5; %% parameter for attenuation of angles (must be odd)

sigma_edge = 1;

angle_step = 2 * pi / num_angles;
angles = 0:angle_step:2*pi;
angles(num_angles+1) = []; % bin centers

[hgt wid] = size(I);

[G_X,G_Y]=gen_dgauss(sigma_edge);
I_X = filter2(G_X, I, 'same'); % vertical edges
I_Y = filter2(G_Y, I, 'same'); % horizontal edges
I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
I_theta = atan2(I_Y,I_X);
%I_theta(find(isnan(I_theta))) = 0; % necessary???? AS: no :)
dipshow(I_mag);
dipshow(I_theta);


% grid 
grid_x = patch_size/2:grid_spacing:wid-patch_size/2+1;
grid_y = patch_size/2:grid_spacing:hgt-patch_size/2+1;

% make orientation images
I_orientation = zeros([hgt, wid, num_angles], 'single');

% for each histogram angle
cosI = cos(I_theta);
sinI = sin(I_theta);
for a=1:num_angles
    % compute each orientation channel
    tmp = (cosI*cos(angles(a))+sinI*sin(angles(a))).^alpha;
    tmp = tmp .* (tmp > 0);

    % weight by magnitude
    I_orientation(:,:,a) = tmp .* I_mag;
end

dipshow(I_orientation);

% Convolution formulation:
weight_kernel = zeros(patch_size,patch_size);
r = patch_size/2;
cx = r - 0.5;
sample_res = patch_size/num_bins;
weight_x = abs((1:patch_size) - cx)/sample_res;
weight_x = (1 - weight_x) .* (weight_x <= 1);
%weight_kernel = weight_x' * weight_x;

for a = 1:num_angles
    %I_orientation(:,:,a) = conv2(I_orientation(:,:,a), weight_kernel, 'same');
    I_orientation(:,:,a) = conv2(weight_x, weight_x', I_orientation(:,:,a), 'same');
end

dipshow(log10(1e-5 + I_orientation));

figure()
tiledlayout(2,5,"TileSpacing","compact")
for i = 1:4
    nexttile
    imshow(I_orientation(:,:,i))
    xticks('manual')
    yticks('manual')
    colormap gray
end
nexttile
imshow(I)
xticks('manual')
yticks('manual')
colormap gray

for i = 5:8
    nexttile
    imshow(I_orientation(:,:,i))
    xticks('manual')
    yticks('manual')
    colormap gray
end

% Sample SIFT bins at valid locations (without boundary artifacts)
% find coordinates of sample points (bin centers)
[sample_x, sample_y] = meshgrid(linspace(1,patch_size+1,num_bins+1));
sample_x = sample_x(1:num_bins,1:num_bins); sample_x = sample_x(:)-patch_size/2;
sample_y = sample_y(1:num_bins,1:num_bins); sample_y = sample_y(:)-patch_size/2;

sift_arr = zeros([length(grid_y) length(grid_x) num_angles*num_bins*num_bins], 'single');
b = 0;
for n = 1:num_bins*num_bins
    sift_arr(:,:,b+1:b+num_angles) = I_orientation(grid_y+sample_y(n), grid_x+sample_x(n), :);
    b = b+num_angles;
end

color_image = showColorSIFT(sift_arr);
nexttile
imshow(color_image)
xticks('manual')
yticks('manual')

figure
plot(sample_x - 1/2, sample_y + 1/2, Marker = '+', LineStyle = 'None', MarkerSize = 12)
hold on
plot(-1/2, 1/2, Marker = '*', LineStyle = 'None', MarkerSize = 12, MarkerFaceColor = 'r', MarkerEdgeColor = 'r')
%xticks('manual')
%yticks('manual')
grid on
xlim([-4,4])
ylim([-4,4])

