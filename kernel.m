diary('output_log.txt'); 
L=51;
d_values=[0.1,0.5,1,5,10,20,50];
d1=10;
Uxs0=-1.15;
Uys0=0;
Uxs1=1.15;
Uys1=0;
NA=0.25; 
filters=arrayfun(@(d) h(d,L,Uxs0,Uys0,Uxs1,Uys1,NA),d_values,'UniformOutput',false);






obj_name='image_0.tif';
object=double(imread(obj_name));
object= imresize(object, [L L], 'cubic');
object(object<0) = 0;
object(object>255) = 255;
object = object/255;
ux=linspace(-1,1,100);
uy=ux;
[Ux,Uy]=meshgrid(ux,uy);
delta_x=linspace(-25,25,100);
delta_y=delta_x;
[Delta_x,Delta_y]=meshgrid(delta_x,delta_y);

R_U=R(Ux,Uy,Uxs0,Uys0,Uxs1,Uys1,NA);
R_delta=R(Delta_x./sqrt(Delta_x.^2+Delta_y.^2+d1.^2),Delta_y./sqrt(Delta_x.^2+Delta_y.^2+d1.^2),Uxs0,Uys0,Uxs1,Uys1,NA);
x = linspace(-1, 1, size(R_U, 2));
y = linspace(-1, 1, size(R_U, 1));
dx = linspace(-25, 25, size(R_delta, 2));
dy = linspace(-25, 25, size(R_delta, 1));
figure();
subplot(4,3,1);
imagesc(x,y,R_U); 
colormap('gray'); 
colorbar; 

title('R function in Ux Uy'); 
axis xy;
axis image;

subplot(4,3,3);
imagesc(dx,dy,R_delta);
colormap('gray'); 
colorbar; 
title('R function in delta_x delta_y when d=10');
axis xy;
axis image;

subplot(4,3,4);
imshow(object,[]);
title('Original');
colorbar;
for i = 1:length(filters)
    result=conv2(object,filters{i},"same");
    subplot(4,3,i+4);
    imshow(result,[]);
    title(['Filtered with d = ' num2str(d_values(i))]);
    colorbar;
end
saveas(gcf, 'results2.fig');
diary off;   

kernelimage=filters{6};
figure(2);
imshow(kernelimage,[]);
% L=11;
% object=double(imread(obj_name));
% object= imresize(object, [L L], 'cubic');
% object(object<0) = 0;
% object(object>255) = 255;
% object = object/255;
% figure(2);
% for i = 1:length(filters11)
%     result=conv2(object,filters11{i},"same");
%     subplot(1,length(filters11),i);
%     imshow(result,[]);
%     title(['Filtered with d = ' num2str(d_values(i))]);
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
  a=0.06;
  b=0.33;
  theta=0.035;
  nspp=1.06;
  output=(B(Ux,Uy).*(a+b.*P((distance(Ux-Uxs0,Uy-Uys0)-nspp)./theta))-B(Ux,Uy).*(a+b.*P((distance(Ux-Uxs1,Uy-Uys1)-nspp)./theta))).*pupil(Ux,Uy,NA);
end

function output=h(d,L,Uxs0,Uys0,Uxs1,Uys1,NA)
  opts = {'AbsTol',1e-2,'RelTol',1e-2};
  output=zeros(L,L);
  for row=1:L
      for col=1:L
      x_min=col-(L-1)/2-1.5;
      x_max=x_min+1;
      y_min=(L-1)/2-row+0.5;
      y_max=y_min+1;
      %output(row,col) = integral(@(x)integral3(@(y,xp,yp)R((x-xp)./(sqrt((x-xp).^2+(y-yp).^2+d.^2)),(y-yp)./(sqrt((x-xp).^2+(y-yp).^2+d.^2)),Uxs0,Uys0,Uxs1,Uys1,NA),y_min,y_max,-0.5,0.5,-0.5,0.5,opts{:}),x_min,x_max,'ArrayValued',true,opts{:});
      output(row,col) = integral2(@(y,x)R((x)./(sqrt((x).^2+(y).^2+d.^2)),(y)./(sqrt((x).^2+(y).^2+d.^2)),Uxs0,Uys0,Uxs1,Uys1,NA),y_min,y_max,x_min,x_max,'Method', 'iterated', 'AbsTol', 1e-2, 'RelTol', 1e-2);
      end
  end
end