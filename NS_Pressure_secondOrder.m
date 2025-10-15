%-------------Solver for NS Equations with pressure 2nd order

clear all, close all, format long
%% Parameters

Nx=64;   X1=0; X2=2*pi;
Ny=64;   Y1=0; Y2=2*pi;

tend=400;  dt=1e-3;  Nt=round(tend/dt);
Re=100;

x = repmat((X2-X1)*[0:Nx-1]'/Nx+X1, 1, Ny);
y = repmat((Y2-Y1)*[0:Ny-1]/Ny +Y1, Nx, 1);
%--------wave number
kx = repmat(2*pi/(X2-X1)*[0:Nx/2 -Nx/2+1:-1]', 1, Ny);
ky = repmat(2*pi/(Y2-Y1)*[0:Ny/2 -Ny/2+1:-1] , Nx, 1);
%--------laplacian
K2 = (-kx.^2-ky.^2);
K2p = K2;  K2p(1,1)=-1;

%% de-aliasing operator
dealias=zeros(size(x));
dealias([1:ceil(Nx/3) ceil(2*Nx/3)+1:Nx],[1:ceil(Ny/3) ceil(2*Ny/3)+1:Ny])=1;

%% Initial condition
%-----soluzione iniziale----
%random
u =  rand(Nx,Ny);
v = rand(Nx,Ny);
p =  rand(Nx,Ny);



uf1 = fftn(u); uf2=uf1;
vf1 = fftn(v); vf2=vf1;
pf1 = fftn(p);
pf2 = pf1;
%%  Solution
for n = 1:Nt-1
 %per avere un metodo accurato al secondo ordine
    ufe = 2*uf1-uf2;
    vfe = 2*vf1-vf2;
    pfe = 2*pf1-pf2;
    ue = real(ifftn(ufe));
    ve = real(ifftn(vfe));
%-----gradU---------------------    
    ux = real(ifftn(1i*kx.*ufe));
    uy = real(ifftn(1i*ky.*ufe));
    vx = real(ifftn(1i*kx.*vfe));
    vy = real(ifftn(1i*ky.*vfe));
%-----non linear term-----------
    uuxf = fftn(ue.*ux);
    vuyf = fftn(ve.*uy);
    uvxf = fftn(ue.*vx);
    vvyf = fftn(ve.*vy);
%-----predicted solution--------  
    unf = (2*uf1-1/2*uf2 +dt*(-1i*kx.*pfe-uuxf-vuyf))./(3/2-dt*K2/Re);
    vnf = (2*vf1-1/2*vf2 +dt*(-1i*ky.*pfe-uvxf-vvyf))./(3/2-dt*K2/Re);
%-----correction ---------------   
%laplaciano di p(n)= laplaciano di p*(n) + divergenza di u*(n)/dt
    pf2 = pf1;
    pf1 = dealias.*(pfe + (1i*kx.*unf+1i*ky.*vnf)./(dt*K2p));
%    pf1 = dealias.*(pnf + (1i*kx.*unf+1i*ky.*vnf)./(dt*K2p));
    pf1(1,1) = 0;
%-----updating the solution  
    uf2 = uf1;
    vf2 = vf1;
    uf1 = dealias.*(unf - 2/3*dt*1i*kx.*(pf1-pfe));
    vf1 = dealias.*(vnf - 2/3*dt*1i*ky.*(pf1-pfe));
    
    u1=real(ifftn(uf1)); v1=real(ifftn(vf1));
    u1(1,:)=0; u1(Nx,:)=0;
    v1(:,1)=0; v1(:,Ny)=0;
    uf1=fftn(u1); vf1=fftn(v1);
    
    t(n) = (n-1)*dt;
    div(n)=max(max(abs(ux+vy)));
%-----plotting    
    if mod(n,1000)==0 
        figure(1)
        contour(x,y,vx-uy,50),axis image, colormap bluewhitered,
        axis([X1 X2 Y1 Y2]), 
        set(gca,'fontsize',18),colorbar, xlabel x, ylabel y, title([' t=',num2str(n*dt)]), drawnow
        disp([' div=',num2str(max(max(abs(ux+vy))))])
        figure(2)
        plot(t,div), xlabel t, ylabel divU;
    end
end
