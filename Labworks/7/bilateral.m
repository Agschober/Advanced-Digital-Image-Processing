function out = bilateral(in, sigma_t, sigma_s)

% input image dims
[nx, ny] = size(in);

out = zeros(nx,ny);

x_cords = xx(nx ,ny ,'corner' );
y_cords = yy(nx ,ny ,'corner' );
x_cords = im2mat(x_cords);
y_cords = im2mat(y_cords);

for i = 1 : nx
    for j = 1 : ny
        
        distance = (x_cords - i + 1).^2 + (y_cords - j + 1).^2;
        spatial_weights = exp(-distance/(2*sigma_s^2));
        tonal_weights = exp(-(in - in(j,i)).^2/(2*sigma_t^2));
        
        weights = spatial_weights.*tonal_weights;

        weighted_image = weights.*in;

        out(j,i) = sum(weighted_image,'all')/(sum(weights, 'all'));
        
    end
end

% TODO 1
