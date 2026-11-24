#pragma once
#include "Newmark.hpp"
namespace ASC_ode {
inline void SolveODE_Verlet(double end,int n,VectorView<> x,VectorView<> v,
    std::shared_ptr<NonlinearFunction> rhs,std::function<void(double,VectorView<>)> observe={}) {
  if(n<=0) throw std::invalid_argument("Expected positive step count");
  const double h=end/n;
  Vector<> acceleration(x.size());
  rhs->evaluate(x,acceleration);
  for(int i=0;i<n;i++) {
    v+=(h/2)*acceleration;
    x+=h*v;
    rhs->evaluate(x,acceleration);
    v+=(h/2)*acceleration;
    if(observe) observe((i+1)*h,x);
  }
}
}
