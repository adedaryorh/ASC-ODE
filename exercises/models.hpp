#pragma once
#include "timestepper.hpp"
#include <cmath>
#include <numbers>

namespace ASC_ode {
class UnitSpring : public NonlinearFunction {
public:
  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  void evaluate(VectorView<> x, VectorView<> f) const override {
    f(0)=x(1); f(1)=-x(0);
  }
  void evaluateDeriv(VectorView<> x, MatrixView<double> j) const override {
    j=0.; j(0,1)=1; j(1,0)=-1;
  }
};

// State = (capacitor voltage, time). The clock makes the system autonomous.
class Circuit : public UnitSpring {
  double rc;
  static constexpr double omega=100*std::numbers::pi;
public:
  explicit Circuit(double timeConstant) : rc(timeConstant) {}
  void evaluate(VectorView<> x, VectorView<> f) const override {
    f(0)=(std::cos(omega*x(1))-x(0))/rc;
    f(1)=1;
  }
  void evaluateDeriv(VectorView<> x, MatrixView<double> j) const override {
    j=0.; j(0,0)=-1/rc; j(0,1)=-omega*std::sin(omega*x(1))/rc;
  }
  double exact(double t) const {
    double a=omega*rc;
    return (std::cos(omega*t)+a*std::sin(omega*t)-std::exp(-t/rc))/(1+a*a);
  }
};
}
