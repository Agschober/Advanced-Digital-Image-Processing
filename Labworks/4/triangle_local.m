% draw a right-angled triangle with angles of alpha and 90-alpha
function out = triangle_local(alpha)

    if alpha > pi/2 || alpha < 0
        error('Input angle (in radian) should be between 0 and pi/2');
    end
    sz = [256 256];
    out = drawpolygon( newim(sz), [75 75; 181 75; 75 75+156*tan(alpha); 75 75],1); 
    out = fillholes(out)*128;
end