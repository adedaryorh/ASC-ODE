#pragma once
#include "autodiff.hpp"
#include "models.hpp"

namespace ASC_ode {
template<class T> T legendre(int n,T x) {
  T previous(1), current=x;
  if(n==0) return previous;
  for(int k=2;k<=n;k++) {
    T next=(T(2*k-1)*x*current-T(k-1)*previous)/T(k);
    previous=current;
    current=next;
  }
  return current;
}
class PendulumAD : public UnitSpring {
  double frequencySquared;
  template<class T> std::array<T,2> rhs(T angle,T velocity) const {
    return {velocity,T(-frequencySquared)*sin(angle)};
  }
public:
  explicit PendulumAD(double gOverLength=1) : frequencySquared(gOverLength) {}
  void evaluate(VectorView<> x,VectorView<> f) const override {
    auto values=rhs(x(0),x(1));
    f(0)=values[0]; f(1)=values[1];
  }
  void evaluateDeriv(VectorView<> x,MatrixView<double> j) const override {
    auto values=rhs<AutoDiff<2>>(Variable<0>(x(0)),Variable<1>(x(1)));
    for(int row=0;row<2;row++)
      for(int col=0;col<2;col++) j(row,col)=values[row].deriv()[col];
  }
};
}
