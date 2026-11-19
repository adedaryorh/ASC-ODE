#ifndef NEWMARK_HPP
#define NEWMARK_HPP

#include <functional>

#include <Newton.hpp>


// Newmark and generalized alpha:
// https://miaodi.github.io/finite%20element%20method/newmark-generalized/

namespace ASC_ode
{
  // Newmark method for  mass(x'') = rhs(x).
  // The unknown in every step is the new acceleration, the new position and
  // velocity are expressions of the function algebra in it.
  inline void SolveODE_Newmark(double tend, int steps,
                               VectorView<double> x, VectorView<double> dx,
                               std::shared_ptr<NonlinearFunction> rhs,
                               std::shared_ptr<NonlinearFunction> mass,
                               std::function<void(double,VectorView<double>)> callback = nullptr)
  {
    double dt = tend/steps;
    double gamma = 0.5;
    double beta = 0.25;

    Vector<> a(x.size());
    Vector<> v(x.size());

    auto xold = std::make_shared<ConstantFunction>(x);
    auto vold = std::make_shared<ConstantFunction>(dx);
    auto aold = std::make_shared<ConstantFunction>(x);
    rhs->evaluate (xold->get(), aold->get());   // initial acceleration, assumes mass = identity
    a = aold->get();                            // initial guess for Newton's method

    auto anew = std::make_shared<IdentityFunction>(a.size());
    auto vnew = vold + dt*((1-gamma)*aold+gamma*anew);
    auto xnew = xold + dt*vold + dt*dt/2 * ((1-2*beta)*aold+2*beta*anew);

    auto equ = Compose(mass, anew) - Compose(rhs, xnew);

    double t = 0;
    for (int i = 0; i < steps; i++)
      {
        NewtonSolver (equ, a);
        xnew -> evaluate (a, x);
        vnew -> evaluate (a, v);

        xold->set(x);
        vold->set(v);
        aold->set(a);
        t += dt;
        if (callback) callback(t, x);
      }
    dx = v;
  }


  // Generalized alpha method for  mass(x'') = rhs(x).
  // rhoinf in [0,1] controls the damping of high frequencies, rhoinf = 1 means no damping.
  inline void SolveODE_Alpha (double tend, int steps, double rhoinf,
                              VectorView<double> x, VectorView<double> dx, VectorView<double> ddx,
                              std::shared_ptr<NonlinearFunction> rhs,
                              std::shared_ptr<NonlinearFunction> mass,
                              std::function<void(double,VectorView<double>)> callback = nullptr)
  {
    double dt = tend/steps;
    double alpham = (2*rhoinf-1)/(rhoinf+1);
    double alphaf = rhoinf/(rhoinf+1);
    double gamma = 0.5-alpham+alphaf;
    double beta = 0.25 * (1-alpham+alphaf)*(1-alpham+alphaf);

    Vector<> a(x.size());
    Vector<> v(x.size());

    auto xold = std::make_shared<ConstantFunction>(x);
    auto vold = std::make_shared<ConstantFunction>(dx);
    auto aold = std::make_shared<ConstantFunction>(ddx);

    auto anew = std::make_shared<IdentityFunction>(a.size());
    auto vnew = vold + dt*((1-gamma)*aold+gamma*anew);
    auto xnew = xold + dt*vold + dt*dt/2 * ((1-2*beta)*aold+2*beta*anew);

    auto equ = Compose(mass, (1-alpham)*anew+alpham*aold) - (1-alphaf)*Compose(rhs,xnew) - alphaf*Compose(rhs, xold);

    double t = 0;
    a = ddx;

    for (int i = 0; i < steps; i++)
      {
        NewtonSolver (equ, a);
        xnew -> evaluate (a, x);
        vnew -> evaluate (a, v);

        xold->set(x);
        vold->set(v);
        aold->set(a);
        t += dt;
        if (callback) callback(t, x);
      }
    dx = v;
    ddx = a;
  }
}

#endif // NEWMARK_HPP
