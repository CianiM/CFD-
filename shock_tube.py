import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as anim


Data = {'1': { 'initial_solution' : [1.0, 0.75, 1.0, 0.125, 0.0, 0.1],
               'x0' : 0.3,
               'Tmax': 0.2},           
        '2': { 'initial_solution' : [1.0, -2.0, 0.4, 1.0, 2.0, 0.4],
               'x0' : 0.5,
               'Tmax': 0.15},
        '3': { 'initial_solution' : [1.0, 0.0, 1000.0, 1.0, 0.0, 0.01],
               'x0' : 0.5,
               'Tmax': 0.012},
        '4': { 'initial_solution' : [5.99924, 19.5975, 460.894, 5.9942, -6.19633, 46.0950],
               'x0' : 0.4,
               'Tmax': 0.035},
        '5': { 'initial_solution' : [1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01],
               'x0' : 0.8,
               'Tmax': 0.012}}
print("#========================#")
print("#       SHOCK TUBE       #")
print("#========================#")
print("insert test: 1, 2, 3, 4, 5,")
Test = str(input("Test: ")) 
print("roe1 = roe method1, roe2 = roe method2, sw = steger & warming, vl = van leer")
scheme = str(input("scheme: "))
if scheme == 'vl':
    CFL = 0.6
else: 
    CFL = 0.9
#=============================#
q0 = Data[Test]['initial_solution'] 
x0 = Data[Test]['x0']
Tmax = Data[Test]['Tmax']
#=============================#
paradic = dict()
#=============================#
paradic['grid'] = dict()
paradic['test'] = dict()
paradic['initial']= dict()
paradic['solution']=dict()
#=============================#
paradic['grid']['L'] = 1.0
paradic['grid']['Nx']= 100
paradic['grid']['x0']= x0
paradic['grid']['gc']= 1
#=============================#
paradic['initial']['gamma'] = 1.4
paradic['initial']['rhol'] = q0[0]
paradic['initial']['ul']   = q0[1]
paradic['initial']['pl']   = q0[2]
paradic['initial']['rhor'] = q0[3]
paradic['initial']['ur']   = q0[4]
paradic['initial']['pr']   = q0[5]
#=============================#
paradic['solution']['equation'] = 3
paradic['solution']['Tmax'] = Tmax
paradic['solution']['rk'] = 2
paradic['solution']['CFL']= CFL
paradic['solution']['itstamp'] = 10
#====================================
def make_grid(paradic):
    L = paradic['grid']['L'] 
    Nx= paradic['grid']['Nx']
    gc= paradic['grid']['gc']
    x0= paradic['grid']['x0']
    x = np.linspace(0,L,Nx)
    grid_points = np.arange(0,Nx)
    internal_points = np.arange(0+gc,Nx-gc)
    return Nx,L,x, x0, grid_points, internal_points

def initial_solution(paradic):
    Nx,L,x, x0, grid_points, internal_points = make_grid(paradic)
    rhol = paradic['initial']['rhol']
    ul = paradic['initial']['ul']
    pl = paradic['initial']['pl']
    rhor = paradic['initial']['rhor']
    ur = paradic['initial']['ur']
    pr = paradic['initial']['pr']
    rho = np.zeros(Nx)
    u = np.zeros(Nx)
    p = np.zeros(Nx)
    for i in range(Nx):
        if x[i] <= x0:
            rho[i] = rhol; u[i] = ul; p[i] = pl
        else:
            rho[i] = rhor; u[i] = ur; p[i] = pr
    return rho,u,p

def variables(q,paradic):
    gamma = paradic['initial']['gamma']
    gamma1 = ( gamma - 1.0 )
    rho = q[0,:]
    u = q[1,:]/q[0,:]
    p = gamma1*(q[2,:] - 0.5*q[1,:]*q[1,:]/q[0,:])
    a = np.sqrt(gamma*p/rho)
    M = u/a
    e = p/(gamma1*rho)
    e0 = 0.5*rho*u*u + rho*e
    H = (e0 + p)/rho
    return rho,u,p,a,e,e0,H,M
 
def time_step(Nx,dx,CFL,scheme,u,a,M,gamma,gamma1):
    
    s1 = abs(u) - a 
    s2 = abs(u)
    s3 = abs(u) + a 
    s1_max = max(s1[1:Nx-1])
    s2_max = max(s2[1:Nx-1])
    s3_max = max(s3[1:Nx-1])
    s_max = max(s1_max,s2_max,s3_max)
    if scheme == 'vl':
        dt_i = np.zeros(Nx)
        for i in np.arange(1,Nx-1):
            CFL_i = (2*gamma + (abs(M[i]))*(3-gamma))/(gamma+3)
            #dt_i[i] = CFL_i*dx/max(s1[i],s2[i],s3[i])
            dt_i[i] = CFL_i*dx/s3[i]
        dt = min(dt_i[1:Nx-1])
    else:
        dt = CFL*dx/s_max 
    return dt
def roe_average(rhol,rhor,ul,ur,Hl,Hr,gc,Nx,gamma1):
    rholt = np.sqrt(rhol)
    rhort = np.sqrt(rhor)
    rhot = rholt*rhort
    d = rholt + rhort
    ut = (rholt*ul + rhort*ur)/d
    Ht = (rholt*Hl + rhort*Hr)/d
    at = np.sqrt(gamma1*(Ht-0.5*ut*ut))
    return rhot,ut,Ht,at

def eigenvalues(gc,Nx,gamma1,rhol,rhor,ul,ur,al,ar,Hl,Hr,e_fix):
    rhot,ut,Ht,at = roe_average(rhol,rhor,ul,ur,Hl,Hr,gc,Nx,gamma1)
    lambda1 = abs(ut - at)
    lambda2 = abs(ut)
    lambda3 = abs(ut + at)
    if e_fix == 1:
        epp1 = np.zeros(Nx)
        epp2 = np.zeros(Nx)
        epp3 = np.zeros(Nx)
        for i in np.arange(gc,Nx):
            epp2[i] = max(1e-20, (ut[i] - ul[i]), (ur[i] - ut[i]))
            epp3[i] = max(1e-20, ((ut[i] + at[i]) - (ul[i] + al[i])), ((ur[i] + ar[i]) - ( ut[i] + at[i])) )
            epp1[i] = max(1e-20, ((ut[i] - at[i]) - (ul[i] - al[i])), ((ur[i] - ar[i]) - ( ut[i] - at[i])) )
        for i in np.arange(gc,Nx):
            if lambda1[i] <= epp1[i]:
                lambda1[i] = 0.5*(lambda1[i]*lambda1[i]/epp1[i] + epp1[i])
            if lambda2[i] <= epp2[i]:
                lambda2[i] = 0.5*(lambda2[i]*lambda2[i]/epp2[i] + epp2[i])
            if lambda3[i] <= epp3[i]:
                lambda3[i] = 0.5*(lambda3[i]*lambda3[i]/epp3[i] + epp3[i])
    return lambda1,lambda2,lambda3

def roe_flux(eq,gc,Nx,gamma1,e_fix,q):
    gamma = gamma1 +1.
    rhor,ur,pr,ar,er,e0r,Hr,Mr = variables(q,paradic)       
    Fp = np.array([rhor*ur,rhor*ur*ur+pr,ur*((gamma/gamma1)*pr + 0.5*rhor*ur*ur)])
    Fm = np.ones((eq,Nx))
    qm = np.ones((eq,Nx))
    qm[:,gc:] = q[:,0:Nx-1]      
    rhol,ul,pl,al,el,e0l,Hl,Ml = variables(qm,paradic)       
    Fm = np.array([rhol*ul,rhol*ul*ul+pl,ul*((gamma/gamma1)*pl + 0.5*rhol*ul*ul)])
    du1 = (rhor - rhol)
    du2 = (rhor*ur - rhol*ul) 
    du3 = (e0r - e0l)
    rhot,ut,Ht,at= roe_average(rhol,rhor,ul,ur,Hl,Hr,gc,Nx,gamma1)    
    a2 = at*at
    u2 = ut*ut
    alpha1 = np.zeros(Nx)
    alpha2 = np.zeros(Nx)
    alpha3 = np.zeros(Nx)
    lambda1,lambda2,lambda3 = eigenvalues(gc,Nx,gamma1,rhol,rhor,ul,ur,Hl,Hr,al,ar,e_fix)
    Phi=np.zeros((3,Nx))
    for i in np.arange(gc,Nx):
        l1 = lambda1[i]; l2 = lambda2[i]; l3 = lambda3[i];
        u = ut[i]
        a = at[i]
        h = Ht[i]
        alph1=(gamma1)*u*u/(2*a*a);
        alph2=(gamma1)/(a*a);
        wdif = q[:,i]-q[:,i-1];        

        Pinv = np.array([[0.5*(alph1+u/a), -0.5*(alph2*u+1/a),  alph2/2],
                        [1-alph1,                alph2*u,                -alph2 ],
                        [0.5*(alph1-u/a),  -0.5*(alph2*u-1/a),  alph2/2]]);
                
        P    = np.array([[ 1,              1,              1              ],
                        [u-a,        u,           u+a      ],
                        [h-a*u,   0.5*u*u,  h+a*u ]]);
        lamb = np.array([[ abs(u-a),  0,              0                 ],
                        [0,                 abs(u),      0                 ],
                        [0,                 0,              abs(u+a)    ]]);
        

        A=np.dot(P,lamb)
        A=np.dot(A,Pinv)       
        Phi[:,i]=np.dot(A,wdif)
    Ft = np.zeros((eq,Nx))
    Ft[:,gc:] = 0.5*(Fp[:,gc:]+Fm[:,gc:] - Phi[:,gc:])
       
    return Ft
   
def roe_flux2(eq,gc,Nx,gamma1,e_fix,q):
    gamma = gamma1 +1.
    Fp = np.ones((eq,Nx))
    rhor,ur,pr,ar,er,e0r,Hr,Mr = variables(q,paradic)       
    Fp = np.array([rhor*ur,rhor*ur*ur+pr,ur*((gamma/gamma1)*pr + 0.5*rhor*ur*ur)])
    Fm = np.ones((eq,Nx))
    qm = np.ones((eq,Nx))
    qm[:,gc:] = q[:,0:Nx-1]      
    rhol,ul,pl,al,el,e0l,Hl,Ml = variables(qm,paradic)       
    du1 = (rhor - rhol)
    du2 = (rhor*ur - rhol*ul) 
    du3 = (e0r - e0l)
    drho = rhor - rhol
    du = ur - ul
    dp = pr - pl
    rho,u,H,a= roe_average(rhol,rhor,ul,ur,Hl,Hr,gc,Nx,gamma1) 
    lambda1,lambda2,lambda3 = eigenvalues(gc,Nx,gamma1,rhol,rhor,ul,ur,Hl,Hr,al,ar,e_fix)
    
    rm1 = abs(u)
    rm2 = abs(u+a)
    rm3 = abs(u-a)   
    
    u2 = u*u
    a2 = a*a
    alp1 = (drho-dp/a2)*rm1
    alp2 = (dp/(2.0*a2) + rho*du/(2.0*a))*rm2
    alp3 = (dp/(2.0*a2) - rho*du/(2.0*a))*rm3
    df1 = alp1 + alp2 + alp3
    df2 = (u*alp1)+(u+a)*alp2+(u-a)*alp3
    df3 = (0.5*u2*alp1)+(H+a*u)*alp2+(H-u*a)*alp3
    c1 = abs(u-a)*(1/(2*a2))*(dp - a*rho*du)
    c2 = abs(u)*(drho - 1/a2*dp)
    c3 = abs(u+a)*(1/(2*a2))*(dp + a*rho*du)
    '''
    c1 = lambda1*(1/(2*a2))*(dp - a*rho*du)
    c2 = lambda2*(drho - 1/a2*dp)
    c3 = lambda3*(1/(2*a2))*(dp + a*rho*du)
    '''
    phi1 = c1 + c2 + c3
    phi2 = (u-a)*c1 + u*c2 + ( u+a)*c3
    phi3 = (H-u*a)*c1 + u2/2*c2 + (H+u*a)*c3
    Ft = np.zeros((eq,Nx))    
    alpha2 = (gamma1/a2)*( du1*(H- u2 ) + u*du2 - du3 )
    alpha1 = ( 1/(2*a)*( du1*(u + a) - du2 - a*alpha2) )
    alpha3 = (du1 - alpha1 - alpha2)
    df = np.array([df1,df2,df3])
    dphi = np.array([phi1,phi2,phi3])
    # eighenvectors
    K1 = np.array([np.ones(Nx),
                  u-a,
                  H-u*a])
    K2 = np.array([np.ones(Nx),
                  u,
                  0.5*u*u])
    K3 = np.array([np.ones(Nx),
                  u+a,
                  H+u*a])
    Ft = np.zeros((eq,Nx))
    dflux = alpha1*abs(u-a)*K1 + alpha2*abs(u)*K2 + alpha3*abs(u+a)*K3
# aggiorno i flussi 
    Ft[:,gc:] = 0.5*(Fp[:,gc:]+Fp[:,:-1] - df[:,gc:])
    #Ft[:,gc:] = 0.5*(Fp[:,gc:]+Fp[:,:-1] - dphi[:,gc:])
    #Ft[:,gc:] = 0.5*(Fp[:,gc:]+Fp[:,:-1] - dflux[:,gc:])
    return Ft    
def sw_flux(eq,gc,Nx,gamma,gamma1,rho,u,a,H):
    lambda1 = ( u - a )
    lambda2 = ( u ) 
    lambda3 = ( u + a )
    lambda1p = 0.5*(lambda1 + abs(lambda1))
    lambda1m = 0.5*(lambda1 - abs(lambda1))
    lambda2p = 0.5*(lambda2 + abs(lambda2))
    lambda2m = 0.5*(lambda2 - abs(lambda2))
    lambda3p = 0.5*(lambda3 + abs(lambda3))
    lambda3m = 0.5*(lambda3 - abs(lambda3))
    k = rho/(2*gamma)
    Fp = np.zeros((eq,Nx))
    Fm = np.zeros((eq,Nx))
    F  = np.zeros((eq,Nx))
    Fp[0,:] = k*( lambda1p + 2*gamma1*lambda2p + lambda3p )
    Fm[0,:] = k*( lambda1m + 2*gamma1*lambda2m + lambda3m )
    Fp[1,:] = k*( lambda1p*( u - a ) + 2*gamma1*u*lambda2p + ( u + a )*lambda3p )
    Fm[1,:] = k*( lambda1m*( u - a ) + 2*gamma1*u*lambda2m + ( u + a )*lambda3m )
    Fp[2,:] = k*( lambda1p*( H - u*a ) + gamma1*u*u*lambda2p + ( H + u*a )*lambda3p )
    Fm[2,:] = k*( lambda1m*( H - u*a ) + gamma1*u*u*lambda2m + ( H + u*a )*lambda3m )
    for i in np.arange(0,Nx-gc):
        F[:,i] = Fp[:,i] + Fm[:,i+1]
    return F

def vanleer_flux(eq,gc,Nx,gamma,gamma1,rho,u,a,M):
    fmasp = ( 0.25)*rho*a*( 1 + M )*( 1 + M )
    fmasm = (-0.25)*rho*a*( 1 - M )*( 1 - M )
    fmomp = fmasp*(2.0*a/gamma)*(gamma1*0.5*M + 1 )
    fmomm = fmasm*(2.0*a/gamma)*(gamma1*0.5*M - 1 )
    c = ( gamma*gamma/(2*(gamma*gamma - 1)) )
    a2 = a*a
    gamma2 = ( gamma*gamma - 1 )
    kp = ( gamma1/2*M + 1 )
    km = ( gamma1/2*M - 1 )
    fenep = c*fmomp*fmomp/fmasp 
    fenem = c*fmomm*fmomm/fmasm
    Fp = np.zeros((eq,Nx))
    Fm = np.zeros((eq,Nx))
    F  = np.zeros((eq,Nx))
    Fp[0,:] = fmasp
    Fp[1,:] = fmasp*2*a/gamma*kp
    Fp[2,:] = fmasp*2*a2/gamma2*kp*kp
    Fm[0,:] = fmasm
    Fm[1,:] = fmasm*2*a/gamma*km
    Fm[2,:] = fmasm*2*a2/gamma2*km*km
    for i in np.arange(0,Nx-gc):
        F[:,i] = Fp[:,i] + Fm[:,i+1]
    return F

def run(paradic,scheme):
    Nx,L,x, x0, grid_points, internal_points = make_grid(paradic)
    rho,u,p = initial_solution(paradic)
    gc = paradic['grid']['gc']
    eq = paradic['solution']['equation']
    Tmax = paradic['solution']['Tmax']
    itstamp = paradic['solution']['itstamp']
    rk = paradic['solution']['rk']
    CFL = paradic['solution']['CFL']
    gamma = paradic['initial']['gamma']
    gamma1 = ( gamma - 1.0 )
    dx = x[1] - x[0]
    a = np.sqrt(gamma*p/rho)
    M = u/a
    e = p/(gamma1*rho)
    e0 = 0.5*rho*u*u + rho*e
    H = (e0 + p)/rho
    qpr= np.ones((eq,Nx))
    qtmp= np.ones((eq,Nx))
    q = np.array([rho,rho*u,e0])
    t = 0
    dtgl = 1e10 
    if rk == 1:
        rungek = [1]
        ak = 1
    elif rk == 2:
        rungek = [ 1/4 , 3/4]
        #rungek = [ 0.42, 1.0]
    elif rk == 3:
        rungek = [1/6,1/6,4/6]
    elif rk == 4:
        rungek = [1/8,1/8,1/8,5/8]  
        rungek = [4/6,8/6,8/6,4/6]
        rungek = [1/4, 1/3, 1/2, 1]
    it = 0
    dt_i = np.zeros(Nx)
    print("inizio calcolo")
    while t < Tmax:
        print(it)
        if it % itstamp == 0: 
            print("it= ", it, "; t= ", t, ";")
        if it < 5:
            CFL = paradic['solution']['CFL'] 
            CFL = 0.2*CFL
        else:
            CFL = paradic['solution']['CFL']   
        dt_min = time_step(Nx, dx, CFL,scheme, u, a,M,gamma,gamma1)
        dt = min(dtgl,dt_min)
        qold = q
        q[:,0] = q[:,0+gc]
        q[:,Nx-1] = q[:,Nx-1-gc]
# START RUNGEK
        for n in np.arange(rk):
            qpr[0,:] = q[0,:]
            qpr[1,:] = q[1,:]/q[0,:]
            qpr[2,:] = gamma1*(q[2,:]-0.5*q[1,:]*q[1,:]/q[0,:])
            for i in grid_points:
                if rho[i] < 0:
                    print("densita negativa in ", x[i])
                    #exit()                    
              # plt.plot(x,rho)
            for i in grid_points:
                if p[i] < 0:
                    print( " pressione negativa in ", x[i])
                    #exit()
            if scheme == 'roe1':
                e_fix = 1
                flux = roe_flux(eq,gc,Nx,gamma1,e_fix,q)
                df = (flux[:,gc+1:Nx-gc+1]-flux[:,gc:Nx-gc])
                qtmp[:,gc:Nx-gc] = qold[:,gc:Nx-gc] - rungek[n]*(dt/dx)*df
            elif scheme == 'roe2':
                e_fix = 1
                flux = roe_flux2(eq,gc,Nx,gamma1,e_fix,q)
                df = (flux[:,gc+1:Nx-gc+1]-flux[:,gc:Nx-gc])
                qtmp[:,gc:Nx-gc] = qold[:,gc:Nx-gc] - rungek[n]*(dt/dx)*df
            elif scheme == 'sw':
                flux = sw_flux(eq,gc,Nx,gamma,gamma1,rho,u,a,H)
                df = (flux[:,gc:Nx-gc]-flux[:,gc-1:Nx-gc-1])
                qtmp[:,gc:Nx-gc] = qold[:,gc:Nx-gc] - rungek[n]*(dt/dx)*df
            elif scheme == 'vl':
                flux = vanleer_flux(eq, gc, Nx, gamma, gamma1, rho, u, a, M)
                df = (flux[:,gc:Nx-gc]-flux[:,gc-1:Nx-gc-1])
                qtmp[:,gc:Nx-gc] = qold[:,gc:Nx-gc] - rungek[n]*(dt/dx)*df
            qtmp[:,0] = qtmp[:,0+gc]
            qtmp[:,Nx-1] = qtmp[:,Nx-1-gc]
            q = qtmp
            rho,u,p,a,e,e0,H,M = variables(q,paradic)
# EXIT RUNGEK
        it = it+1
        t = t + dt      

    sol_list=[]
    sol_list.append(rho)
    sol_list.append(u)
    sol_list.append(p)
    sol_list.append(e)
    print("Computation done in it= ", it, "; t= ", t, ".")
    return x,q,t,sol_list
                
x,q,t,sol_list = run(paradic,scheme)   
rho,u,p,a,e,e0,H,M = variables(q,paradic)

file = "exact"+Test+".txt"
print(file)

data = np.loadtxt(file)
rho_ex = data[:,1]
u_ex = data[:,2]
p_ex = data[:,3]
gamma = 1.4
gamma1 = gamma -1.0
e_ex = p_ex/(gamma1*rho_ex)
x_ex = np.linspace(0,1,100)
sol_ex=list()
sol_ex.append(rho_ex)
sol_ex.append(u_ex)
sol_ex.append(p_ex)
sol_ex.append(e_ex)
grafici = ['rho','u','p','e']
grafici_ex = ['rho_ex','u_ex','p_ex','e_ex']
names = ['density','velocity','pressure','internal energy']
#==============plot===================
for sol in np.arange(0,len(sol_list)):
    figure1=plt.figure(figsize=[7.0,5.0])
    plt.plot(x_ex,sol_ex[sol],'k-',label= grafici_ex[sol])
    plt.plot(x,sol_list[sol],'ro', label = grafici[sol])
    plt.xlabel('x')
    plt.ylabel(names[sol])
    plt.title("TEST = " +str(Test))
    plt.legend()
plt.show()



    