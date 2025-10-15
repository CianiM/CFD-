%risolve l'equazione del calore 1D con un metodo alla differenze finite 
%accurato al secondo ordine nel tempo, utile per prendere la matrice delle 
%derivate seconde D2  

clear all, close all, format long

Lx=4; Nx=20; 
tend=10; dt=1e-3; Nt=round(tend/dt);
k=1;
x=linspace(0,Lx,Nx); dx=(x(2)-x(1));

D2=(1+dt)*speye(Nx)-k*dt*spdiags(ones(Nx,1)*[1 -2 1]/dx^2,-1 :1,Nx,Nx) ;
D2(1,:)=0; D2(1,1)=1;
D2(Nx,:)=0; D2(Nx,Nx)=1+1/dx; D2(Nx,Nx-1)=-1/dx;
D2i=inv(D2);
T(:,1)=20*ones(Nx,1);
for t=2:Nt
    rhs(:,t)=T(:,t-1)+20*dt;
    rhs(1,t)=127;
    rhs(Nx,t)=20;
    T=D2i*rhs;
    %T(1,t)=127;
    %T(Nx,t)=0;
    plot(x,T(:,t)),drawnow
    title(['t=',num2str((t-1)*dt)]), drawnow
end

    