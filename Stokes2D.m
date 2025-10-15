%risolve equazioni di NavierStokes con un metodo alle differenze finite

clear all, close all, format long

X1 = 0; X2 = 1; Nx = 100; 
Y1 = 0; Y2 = 1; Ny = 100;

tend = 20; dt = 1e-3; Nt = round(tend/dt);
Re = 100;

x = linspace(X1,X2,Nx);                              
y = linspace(Y1,Y2,Ny);

dx = x(2)-x(1);                                     
dy = y(2)-y(1);

Dx  = 0+spdiags(ones(Nx,1)*[-1 0 1]/(2*dx),-1:1,Nx,Nx);
Dy  = 0+spdiags(ones(Ny,1)*[-1 0 1]/(2*dy),-1:1,Ny,Ny);
Dxp  = 0+spdiags(ones(Nx,1)*[-1 0 1]/(2*dx),-1:1,Nx,Nx);
Dyp  = 0+spdiags(ones(Ny,1)*[-1 0 1]/(2*dy),-1:1,Ny,Ny);

D2x = 0+spdiags(ones(Nx,1)*[1 -2 1]/dx^2,-1 :1,Nx,Nx);
D2y = 0+spdiags(ones(Ny,1)*[1 -2 1]/dy^2,-1 :1,Ny,Ny);

 
[Qx, Ex] = eig(D2x) ; invQx=inv(Qx) ;
Ex=repmat(diag(Ex),1,Ny) ;
[Qy, Ey] = eig(D2y) ; invQy=inv(Qy)' ; Qy=Qy' ;
Ey=repmat(diag(Ey)',Nx,1) ;
%condizioni iniziali
U = zeros(Nx,Ny);
V = zeros(Nx,Ny);
P = zeros(Nx,Ny);
Pn= P;   %pn = pressione al time step precedente

U(1,:)=0;
U(Nx,:)=0;
U(:,1)=0;
U(:,Ny)=0;

V(1,:)=0;
V(Nx,:)=0;
V(:,1)=0;
V(:,Ny)=0;

for n=1:Nt
%--------predictor
    Un = U + dt*( (D2x*U + U*D2y')/Re - Dx*Pn - U.*(Dx*U) - V.*(U*Dy') );
    Vn = V + dt*( (D2x*V + V*D2y')/Re - Pn*Dy'- V.*(Dx*V) - V.*(V*Dy') );
    
    %calcolo della pressione aggiornata per rispettare la divergenza
  %  Pfn = invQx*Pn*invQy;    %proiezione nello spazio degli autovalori
    rhs=D2x*Pn+Pn*D2y'+Dxp*U/dt+V*Dyp'/dt;
    R =invQx*rhs*invQy;
    Pf = R./(Ex+Ey); %calcolo
    P = Qx*Pf*Qy; %ritorno nello spazio fisico
 %condizioni al contorno sulla pressione: gradiente nullo alle pareti
    P(1,:)=P(2,:);  P(Nx,:)=P(Nx-1,:);
    P(:,1)=P(:,2);  P(:,Ny)=P(:,Ny-1);
    
    %correction sulla velocità
    U = Un - dt*Dx*( P - Pn );
    V = Vn - dt*( P - Pn )*Dy';
    Pn = P;
    U(1,:)=0;
    U(Nx,:)=0;
    U(:,1)=0;
    U(:,Ny)=1;
    V(1,:)=0;
    V(Nx,:)=0;
    V(:,1)=0;
    V(:,Ny)=0;
    
    if mod(n,100)==0
    figure(1)
    contourf(x,y,U',50), axis image, colormap bluewhitered,
    set(gca,'fontsize',18)
    colorbar
    xlabel x, ylabel y
    title(['t=',num2str(n*dt)]), drawnow
    figure(2)
    contourf(x,y,V',50), axis image, colormap bluewhitered,
    set(gca,'fontsize',18)
    colorbar
    xlabel x, ylabel y
    title(['t=',num2str(n*dt)]), drawnow
    end
end