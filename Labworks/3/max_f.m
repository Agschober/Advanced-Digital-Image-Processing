function out = max_f(in, bsize)

[N, M] = size(in);
out = zeros(N, M);

% check the structuring element size
if ~mod(bsize, 2)
    error('Error. \nStructuring element size should be an odd number.')
end

% define the flat (constant) structuring element
x = (1:bsize)';
y = (1:bsize)';

% increase the image size at the boundaries (zero padding)
inPad = extend(in, [N+bsize-1 M+bsize-1], 'symmetric',0);
inPad = im2mat(inPad); % convert to matlab array for speed with loop


% loop and slide the SE over each pixel of the input image
for i= 1:size(inPad,2)-(bsize-1)
    for j=1:size(inPad,1)-(bsize-1)

        %In the padded input you already have an offset, so you just have
        %to take the slice of the size of the structuring element in all
        %dimensions and find the maximum here (exact same just replace maximum by minimum in min filter)
        out(j,i) = max(inPad(j:j+bsize-1,i:i+bsize-1), [],'all');
        
    end
end
out =mat2im(out); %convert back to dipimage
