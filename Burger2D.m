clear all, close all, format long

Nx=64*4; 
Ny=64*4;
X1=-1; X2=1;
Y1=-1; Y2=1;

Tend=2; dt=1e-3; Nt=round(Tend/dt);
Re=100;
%---------------
x=repmat((X2-X1)*[0:Nx-1]'/Nx+X1,1,Ny);
y=repmat((Y2-Y1)*[0:Nx-1] /Nx+Y1,Nx,1);
kx=repmat((2*pi)/(X2-X1)*[0:Nx/2-1 -Nx/2:-1]',1,Ny);
ky=repmat((2*pi)/(Y2-Y1)*[0:Nx/2-1 -Nx/2:-1] ,Nx,1);
%---------------
dealias=zeros(Nx);
dealias([1:ceil(Nx/3) ceil(2*Nx/3)+1:Nx],[1:ceil(Ny/3) ceil(2*Ny/3)+1:Ny] )=1;




u1=sin(x).*cos(y);
uf1=fftn(u1);
uf2=uf1;
u2=u1;

v1=-cos(x).*sin(y);
vf1=fftn(v1);
vf2=vf1;
v2=v1;
for n=1:Nt
    u = 2*u1-u2;
    v = 2*v1-v2;
    ux = real(ifftn( 1.i*kx.*(2*uf1-uf2) ));
    uy = real(ifftn( 1.i*ky.*(2*uf1-uf2) ));
    vx = real(ifftn( 1.i*kx.*(2*vf1-vf2) ));
    vy = real(ifftn( 1.i*ky.*(2*vf1-vf2) ));
    %ue=2*u1-u0;
    %ve=2*v1-v0;
    %ufe=2*uf1-uf0;
    %vfe=2*vf1-vf0;
    %------gradiente
    %ux=real(ifftn(1i*kx.*ufe));
    %uy=real(ifftn(1i*ky.*ufe));
    %vx=real(ifftn(1i*kx.*vfe));
    %vy=real(ifftn(1i*ky.*vfe));
    %-------
    %unonlin=fftn(ue.*ifftn(1i*kx*ufe)+ve.*ifftn(1i*ky.*ufe));
    %vnonlin=fftn(ue.*ifftn(1i*kx*vfe)+ve.*ifftn(1i*ky.*vfe));
    
    
    %uf2 = (2*uf1/dt-1/2*uf0/dt-fftn(ue.*ux+ve.*uy))./(3/(2*dt)+1/Re*(kx.^2+ky.^2));
    %vf2 = (2*uf1/dt-1/2*uf0/dt-fftn(ue.*vx+ve.*vy))./(3/(2*dt)+1/Re*(kx.^2+ky.^2));
    uf0 = ( (4*uf1-uf2)/(2*dt) - fftn(u.*ux+v.*uy) )./( 3/(2*dt) +(kx.^2+ky.^2)/Re );
    vf0 = ( (4*vf1-vf2)/(2*dt) - fftn(u.*vx+v.*vy) )./( 3/(2*dt) +(kx.^2+ky.^2)/Re );
    
    %uf0=uf1; 
    %vf0=vf1; 
    %u0=u1; 
    %v0=v1; 
    %uf1=dealias.*uf2;
    %vf1=dealias.*vf2;
    %u1 = real(ifftn(uf1));
    %v1 = real(ifftn(vf1));
    uf2=uf1; 
    vf2=vf1; 
    u2=u1; 
    v2=v1; 
    uf1=dealias.*uf0;
    vf1=dealias.*vf0;
    u1 = real(ifftn(uf1));
    v1 = real(ifftn(vf1));

    if mod(n,100)==0
        contourf(x,y,vx-uy,20), axis image, %colormap bluewhitered,
        axis([X1 X2 Y1 Y2]), 
        set(gca,'fontsize',18),colorbar, xlabel x, ylabel y, title([' t=',num2str(n*dt)]), drawnow
    end
end