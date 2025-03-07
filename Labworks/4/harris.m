% harris corner detector

function [xy, R] = harris(img, sigma_grad, sigma_tensor, k, thresh)

    % TODO1
    % compute image derivatives
    imgDx = dx(img, sigma_grad);
    imgDy = dy(img, sigma_grad);
    
    % construct the structure tensor by computing its elements
    S_xx = gaussf(imgDx.*imgDx, sigma_tensor);
    S_xy = gaussf(imgDy.*imgDx, sigma_tensor);
    S_yy = gaussf(imgDy.*imgDy, sigma_tensor);
    
    % compute the cornerness according to Harris measure
    R= S_xx.*S_yy - S_xy.*S_xy - k*(S_xx+S_yy).^2;
    
    
    % find the location of pixels with cornerness above thresh
    xy = findcoord(R>thresh);
    
end
