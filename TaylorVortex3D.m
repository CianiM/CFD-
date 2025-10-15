%%%-------------Solver for NS Equations with pressure 2nd order

clear all, close all, format long
%%%  Parameters

tend=4;  dt=1e-3;  Nt=round(tend/dt);
Re=100;

Nx=64;   X1=0; X2=2*pi;
Ny=64;   Y1=0; Y2=2*pi;
Nz=64;   Z1=0; Z2=2*pi;

x = repmat((X2-X1)*[0:Nx-1]'/Nx+X1, 1, Ny, Nz);
y = repmat((Y2-Y1)*[0:Ny-1]/Ny +Y1, Nx, 1, Nz);
zz = linspace(Z1,Z2,Nz);
for k=1:Nz
    z(:,:,k)=zz(k)*ones(Nx,Ny);
end
%--------wave number
kx = repmat(2*pi/(X2-X1)*[0:Nx/2 -Nx/2+1:-1]', 1, Ny, Nz);
ky = repmat(2*pi/(Y2-Y1)*[0:Ny/2 -Ny/2+1:-1] , Nx, 1, Nz);
kzz=(Z2-Z1)*[0:Nz/2 -Nz/2+1:-1];
for k=1:Nz
    kz(:,:,k)=kzz(k)*ones(Nx,Ny);
end
%--------laplacian
K2 = (-kx.^2-ky.^2-kz.^2);
K2p = K2;  K2p(1,1,1)=-1;


%% de-aliasing operator
dealias=zeros(size(x));
dealias([1:ceil(Nx/3) ceil(2*Nx/3)+1:Nx],[1:ceil(Ny/3) ceil(2*Ny/3)+1:Ny])=1;

%% Initial condition
%-----soluzione iniziale----

u =  sin(x).*cos(y).*cos(z);
v =  -cos(x).*sin(y).*cos(z);
w = zeros(Nx,Ny,Nz);
p =  1/16*(cos(2*x)+cos(2*y)).*(cos(2*z)+2);

uf1 = fftn(u); uf2=uf1;
vf1 = fftn(v); vf2=vf1;
wf1 = fftn(w); wf2=wf1;
pf1 = fftn(p);
pf2 = pf1;
%%  Solution
for n = 1:Nt-1
    
    ufe = 2*uf1-uf2;
    vfe = 2*vf1-vf2;
    wfe = 2*wf1-wf2;
    pfe = 2*pf1-pf2;
    ue = real(ifftn(ufe));
    ve = real(ifftn(vfe));
    we = real(ifftn(wfe));
%-----gradU---------------------    
    ux = real(ifftn(1i*kx.*ufe));
    uy = real(ifftn(1i*ky.*ufe));
    uz = real(ifftn(1i*kz.*ufe));
    vx = real(ifftn(1i*kx.*vfe));
    vy = real(ifftn(1i*ky.*vfe));
    vz = real(ifftn(1i*kz.*vfe));
    wx = real(ifftn(1i*kx.*wfe));
    wy = real(ifftn(1i*ky.*wfe));
    wz = real(ifftn(1i*kz.*wfe));
%-----non linear term-----------
    uuxf = fftn(ue.*ux);
    vuyf = fftn(ve.*uy);
    wuzf = fftn(we.*uz);
    uvxf = fftn(ue.*vx);
    vvyf = fftn(ve.*vy);
    wvzf = fftn(we.*vz);
    uwxf = fftn(ue.*wx);
    vwyf = fftn(ve.*wy);
    wwzf = fftn(we.*wz);
%-----predicted solution--------    
    unf = (2*uf1-1/2*uf2 +dt*(-1i*kx.*pfe-uuxf-vuyf-wuzf))./(3/2-dt*K2/Re);
    vnf = (2*vf1-1/2*vf2 +dt*(-1i*ky.*pfe-uvxf-vvyf-wvzf))./(3/2-dt*K2/Re);
    wnf = (2*wf1-1/2*wf2 +dt*(-1i*kz.*pfe-uwxf-vwyf-wwzf))./(3/2-dt*K2/Re);
%-----correction ---------------    
    pf2 = pf1;
    pf1 = pfe + 3/2*(1i*kx.*unf+1i*ky.*vnf+1i*kz.*wnf)./(dt*K2p);
    pf1(1,1,1) = 0;
%-----updating the solution  
    uf2 = uf1;
    vf2 = vf1;
    wf2 = wf1;
    uf1 = dealias.*(unf - 2/3*dt*1i*kx.*(pf1-pfe));
    vf1 = dealias.*(vnf - 2/3*dt*1i*ky.*(pf1-pfe));
    wf1 = dealias.*(wnf - 2/3*dt*1i*kz.*(pf1-pfe));
    pf2=pf1;
    
    u1=real(ifftn(uf1)); v1=real(ifftn(vf1)); w1=real(ifftn(wf1));
    u1(1,:)=0; u1(Nx,:)=0;
    v1(:,1)=0; v1(:,Ny)=0;
    uf1=fftn(u1); vf1=fftn(v1);
    
    t(n) = (n-1)*dt;
    %div(n)=max(max(abs(ux+vy+wz)));
%-----plotting    
    if mod(n,1000)==0 
        figure(1)
        contour(x,y,vx-uy,30),axis image, colormap bluewhitered,
        axis([X1 X2 Y1 Y2]), 
        set(gca,'fontsize',18),colorbar, xlabel x, ylabel y, title([' t=',num2str(n*dt)]), drawnow
        %disp([' div=',num2str(max(max(abs(ux+vy))))])
        %figure(2)
       % plot(t,div), xlabel t, ylabel divU;
    end
end