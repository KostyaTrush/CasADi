from casadi import *

U = 80.0
A_C = 0.01
L_C = 3
V_P = 1.5
H = 0.18
W = 0.25
a_s = 340
gamma = 0.411
x10 = 0.533
x20 = 0.3
k1_const = 3*H*x20/2/W**2*(x20/W-2)
k2_const = 3*H/2/W**2*(x20/W-1)
k3_const = H/2/W**3
B = U/2/a_s*sqrt(V_P/A_C/L_C)

T = 20. # Time horizon
N = 50 # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1, x2)
u = MX.sym('u')
# Model equations
xdot = vertcat(1/B*(x2 - gamma*(sqrt(x1+x10) - sqrt(x10))), B*(-k3_const*x2**3 - k2_const*x2**2 - k1_const*x2 - x1 - u))

# Objective term
L = x1**2 + x2**2 + u**2

dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
opts = {'tf':T/N}
F = integrator('F', 'cvodes', dae, opts)

# Start with an empty NLP
w = []
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += [0.1, 0]
ubw += [0.1, 0]
w0 += [0.1, 0]

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [0]
    ubw += [0.2]
    w0  += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['xf']
    J=J+Fk['qf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), 2)
    w   += [Xk]
    lbw += [-inf, -inf]
    ubw += [inf,  inf]
    w0  += [0, 0]

    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0, 0]
    ubg += [0, 0]

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# Plot the solution
x1_opt = w_opt[0::3]
x2_opt = w_opt[1::3]
u_opt = w_opt[2::3]

tgrid = [T/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-o')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.ylim(-0.3, 0.25)
plt.savefig('1.png', format='png')
plt.show()