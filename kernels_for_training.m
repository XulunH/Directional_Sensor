kernel_size=9;
d=10;
Uxs0=1.15;
Uys0=0;
Uxs1=-1.15;
Uys1=0;
NA=0.3; 
superpixel_size = 3;
pad_size =ceil((kernel_size - 1) / 2); 

Ux=linspace(-1,1,100);
Uy=Ux;
[UX,UY]=meshgrid(Ux,Uy);

figure(100);
imshow(R(UX,UY,Uxs0,Uys0,Uxs1,Uys1,NA),[]);

screensize = get(0, 'ScreenSize');
screen_width = screensize(3);
screen_height = screensize(4);

theta = pi/4;  
Rotation_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)];
 
if superpixel_size == 2
    filters = zeros(kernel_size,kernel_size,4);
  
    for i = 1:4
        position=i;
        x_relative=-0.5+1/superpixel_size/2+1/superpixel_size*mod(position,superpixel_size);
        y_relative=-0.5+1/superpixel_size/2+1/superpixel_size*floor((position-1)/superpixel_size);
        filters(:, :, i) = h(d, kernel_size,x_relative,y_relative, Uxs0, Uys0, Uxs1, Uys1, NA);
    
        points = [Uxs0 Uxs1; Uys0 Uys1];
        rotated_points = Rotation_matrix * points;
    
        Uxs0 = rotated_points(1, 1);
        Uys0 = rotated_points(2, 1);
        Uxs1 = rotated_points(1, 2);
        Uys1 = rotated_points(2, 2);
        
    end
    global_min = min(filters(:)); 
    global_max = max(filters(:));
    figure();
    for i = 1:4
        subplot(2, 2, i);  
        imshow(filters(:, :, i), [global_min global_max]);
        colorbar;
    end
end

if superpixel_size == 3
    filters = zeros(kernel_size,kernel_size,8);
    
    for i = 1:8
        if i<=4
            position=i;
        else 
            position=i+1;
        end
        x_relative=-0.5+1/superpixel_size/2+1/superpixel_size*mod(position,superpixel_size);
        y_relative=-0.5+1/superpixel_size/2+1/superpixel_size*floor((position-1)/superpixel_size);
        filters(:, :, i) = h(d, kernel_size,x_relative,y_relative, Uxs0, Uys0, Uxs1, Uys1, NA);
    
        points = [Uxs0 Uxs1; Uys0 Uys1];
        rotated_points = Rotation_matrix * points;
    
        Uxs0 = rotated_points(1, 1);
        Uys0 = rotated_points(2, 1);
        Uxs1 = rotated_points(1, 2);
        Uys1 = rotated_points(2, 2);
        
    end
    global_min = min(filters(:)); 
    global_max = max(filters(:));
    figure();
    for i = 1:8
        subplot(2, 4, i);  
        imshow(filters(:, :, i), [global_min global_max]);
        colorbar;
    end
end


save('filters.mat', 'filters');

% obj_name='image_0.tif';
% object=double(imread(obj_name));
% object(object<0) = 0;
% object(object>255) = 255;
% object = object/255;
% 
% figure(1);
% set(gcf, 'Position', [0, screen_height / 2-80, screen_width, screen_height / 2]);
% subplot(1,5,1);
% imshow(object,[]); axis image;
% for i = 1:4
% 
%     result = conv2(object, filters(:, :, i), 'same');
%     result=result/max(result(:));
%     subplot(1,5,i+1);
%     imshow(result,[]);
%     axis image; 
% 
%     title(sprintf('Filter %d Convolution', i));
% end
% object=padarray(object,[pad_size pad_size],'symmetric');
% figure(2);
% set(gcf, 'Position', [0, -35, screen_width, screen_height / 2]);
% subplot(1,5,1);
% imshow(object,[]); axis image;
% for i = 1:4
% 
%     result = conv2(object, filters(:, :, i), 'same');
%     subplot(1,5, i+1);
%     [height, width] = size(result);
% 
%     central_start_row = round((height - 28) / 2) + 1;
%     central_end_row = central_start_row + 28 - 1;
%     central_start_col = round((width - 28) / 2) + 1;
%     central_end_col = central_start_col + 28 - 1;
% 
% 
%     central_region = result(central_start_row:central_end_row, central_start_col:central_end_col);
%     central_region=central_region/max(central_region(:));
% 
%     imshow(central_region, []);
%     axis image; 
% 
%     title(sprintf('Filter %d Convolution', i));
% end
% obj_name='fashion_mnist_2.tif';
% object=double(imread(obj_name));
% object(object<0) = 0;
% object(object>255) = 255;
% object = object/255;
% 
% figure(3);
% set(gcf, 'Position', [0, screen_height / 2-80, screen_width, screen_height / 2]);
% subplot(1,5,1);
% imshow(object,[]); axis image;
% for i = 1:4
% 
%     result = conv2(object, filters(:, :, i), 'same');
%     result=result/max(result(:));
%     subplot(1,5,i+1);
%     imshow(result,[]);
%     axis image; 
% 
%     title(sprintf('Filter %d Convolution', i));
% end
% object=padarray(object,[pad_size pad_size],'symmetric');
% figure(4);
% set(gcf, 'Position', [0, -35, screen_width, screen_height / 2]);
% subplot(1,5,1);
% imshow(object,[]); axis image;
% for i = 1:4
% 
%     result = conv2(object, filters(:, :, i), 'same');
% 
%     subplot(1, 5, i+1);
%     [height, width] = size(result);
% 
% central_start_row = round((height - 28) / 2) + 1;
% central_end_row = central_start_row + 28 - 1;
% central_start_col = round((width - 28) / 2) + 1;
% central_end_col = central_start_col + 28 - 1;
% 
% central_region = result(central_start_row:central_end_row, central_start_col:central_end_col);
% central_region=central_region/max(central_region(:));
% 
% imshow(central_region, []);
% 
%     axis image; 
%     title(sprintf('Filter %d Convolution', i));
% end
function output = B(x, y)
    u = distance(x,y);  
    output = zeros(size(u)); 

    mask = u < 1;
    output(mask) = 1 - u(mask).^40;
end

function output = pupil(x, y, NA)
    u = distance(x,y);  
    output = zeros(size(u)); 
    
    mask = u < NA;
    output(mask) = 1-u(mask).^40;
end


function output=P(x)
   output=1./(1+x.^2);
end

function output=distance(x,y)
  output=sqrt(x.^2+y.^2);
end


function output=R(Ux,Uy,Uxs0,Uys0,Uxs1,Uys1,NA)
  a=0.06; % related to overall background brightness
  b=0.33; % related to brightness of the circle
  theta=0.035; % related to the spread(blurring) of the circle
  nspp=1.06; % related to radius of circle
  output=(B(Ux,Uy).*(a+b.*P((distance(Ux-Uxs0,Uy-Uys0)-nspp)./theta))-B(Ux,Uy).*(a+b.*P((distance(Ux-Uxs1,Uy-Uys1)-nspp)./theta))).*pupil(Ux,Uy,NA);
end

% function output=h(d,kernel_size,Uxs0,Uys0,Uxs1,Uys1,NA)
%   opts = {'AbsTol',1e-2,'RelTol',1e-2};
%   output=zeros(kernel_size,kernel_size);
%   for row=1:kernel_size
%       for col=1:kernel_size
%       x_min=col-(kernel_size-1)/2-1.5;
%       x_max=x_min+1;
%       y_min=(kernel_size-1)/2-row+0.5;
%       y_max=y_min+1;
%       %output(row,col) = integral(@(x)integral3(@(y,xp,yp)R((x-xp)./(sqrt((x-xp).^2+(y-yp).^2+d.^2)),(y-yp)./(sqrt((x-xp).^2+(y-yp).^2+d.^2)),Uxs0,Uys0,Uxs1,Uys1,NA),y_min,y_max,-0.5,0.5,-0.5,0.5,opts{:}),x_min,x_max,'ArrayValued',true,opts{:});
%       output(row,col) = integral2(@(y,x)R((x)./(sqrt((x).^2+(y).^2+d.^2)),(y)./(sqrt((x).^2+(y).^2+d.^2)),Uxs0,Uys0,Uxs1,Uys1,NA),y_min,y_max,x_min,x_max,'Method', 'iterated', 'AbsTol', 1e-2, 'RelTol', 1e-2);
%       end
%   end
% end
function output=h(d,kernel_size,x_relative,y_relative,Uxs0,Uys0,Uxs1,Uys1,NA)
  opts = {'AbsTol',1e-2,'RelTol',1e-2};
  output=zeros(kernel_size,kernel_size);
  for row=1:kernel_size
      for col=1:kernel_size
      x_min=-kernel_size/2+col-1-x_relative;
      x_max=x_min+1;
      y_min=-kernel_size/2+row-1-y_relative;
      y_max=y_min+1;
      %output(row,col) = integral(@(x)integral3(@(y,xp,yp)R((x-xp)./(sqrt((x-xp).^2+(y-yp).^2+d.^2)),(y-yp)./(sqrt((x-xp).^2+(y-yp).^2+d.^2)),Uxs0,Uys0,Uxs1,Uys1,NA),y_min,y_max,-0.5,0.5,-0.5,0.5,opts{:}),x_min,x_max,'ArrayValued',true,opts{:});
      output(row,col) = integral2(@(y,x)R((x)./(sqrt((x).^2+(y).^2+d.^2)),(y)./(sqrt((x).^2+(y).^2+d.^2)),Uxs0,Uys0,Uxs1,Uys1,NA),y_min,y_max,x_min,x_max,'Method', 'iterated', 'AbsTol', 1e-2, 'RelTol', 1e-2);
      end
  end
end
