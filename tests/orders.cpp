#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../exercises/models.hpp"
using namespace ASC_ode;
double springError(TimeStepper& solver,int n) {
  Vector<> y={1,0};
  for(int i=0;i<n;i++) solver.doStep(2*std::numbers::pi/n,y);
  return std::hypot(y(0)-1,y(1));
}
void checkOrder(TimeStepper& solver,int p,int n) {
  const double order=std::log2(springError(solver,n)/springError(solver,2*n));
  REQUIRE(std::abs(order-p)<0.1);
}
TEST_CASE("Spring orders over one revolution") {
  auto rhs=std::make_shared<UnitSpring>();
  ExplicitEuler forward(rhs); ImplicitEuler backward(rhs);
  ImprovedEuler midpoint(rhs); CrankNicolson cn(rhs);
  checkOrder(forward,1,512); checkOrder(backward,1,512);
  checkOrder(midpoint,2,512); checkOrder(cn,2,512);
}
#include "../exercises/functions.hpp"
TEST_CASE("AD derivatives agree with analytic formulas") {
  AutoDiff<1> x=Variable<0>(0.5);
  auto expression=log(exp(x))/cos(x);
  REQUIRE(expression.deriv()[0]==Approx(1/std::cos(.5)+.5*std::sin(.5)/std::pow(std::cos(.5),2)));
  REQUIRE(legendre(3,x).deriv()[0]==Approx((15*.25-3)/2));
  PendulumAD pendulum(9.81);
  Vector<> y={.4,.3}; Matrix<> jac(2,2);
  pendulum.evaluateDeriv(y,jac);
  REQUIRE(jac(0,0)==0); REQUIRE(jac(0,1)==1);
  REQUIRE(jac(1,0)==Approx(-9.81*std::cos(.4))); REQUIRE(jac(1,1)==0);
}
#include "../src/explicitRK.hpp"
TEST_CASE("Collocation and classical RK orders before roundoff dominates") {
  auto rhs=std::make_shared<UnitSpring>();
  auto rk4=classicalRK4(rhs);
  checkOrder(rk4,4,32);
  for(int s=1;s<=3;s++) {
    Vector<> c(s),w(s);
    GaussLegendre(c,w); auto [a,b]=computeABfromC(c);
    ImplicitRungeKutta gauss(rhs,a,b,c);
    checkOrder(gauss,2*s,16);
    GaussRadau(c,w); auto [ar,br]=computeABfromC(c);
    ImplicitRungeKutta radau(rhs,ar,br,c);
    checkOrder(radau,2*s-1,s==1?512:32);
  }
}
#include "../src/adaptive.hpp"
TEST_CASE("Bogacki-Shampine shared stages have orders three and two") {
  auto f=std::make_shared<UnitSpring>();
  BogackiShampine high(f); checkOrder(high,3,32);
  Matrix<> a={{0,0,0,0},{.5,0,0,0},{0,.75,0,0},{2./9,1./3,4./9,0}};
  ExplicitRungeKutta low(f,a,{7./24,.25,1./3,.125},{0,.5,.75,1});
  checkOrder(low,2,32);
}
TEST_CASE("Collocation tableaux reproduce all integrated basis polynomials") {
  for(bool radau : {false,true}) for(int s : {2,3}) {
    Vector<> c(s),w(s);
    if(radau) GaussRadau(c,w); else GaussLegendre(c,w);
    auto [a,b]=computeABfromC(c);
    // Independent reference: integrate each Lagrange cardinal polynomial.
    for(int j=0;j<s;j++) {
      std::vector<double> coefficients{1};
      for(int k=0;k<s;k++) if(k!=j) {
        std::vector<double> next(coefficients.size()+1);
        for(size_t m=0;m<coefficients.size();m++) {
          next[m]-=coefficients[m]*c(k)/(c(j)-c(k));
          next[m+1]+=coefficients[m]/(c(j)-c(k));
        }
        coefficients=next;
      }
      auto integral=[&](double z) {
        double result=0;
        for(size_t m=0;m<coefficients.size();m++)
          result+=coefficients[m]*std::pow(z,m+1)/(m+1);
        return result;
      };
      REQUIRE(b(j)==Approx(integral(1)).margin(1e-13));
      for(int i=0;i<s;i++) REQUIRE(a(i,j)==Approx(integral(c(i))).margin(1e-13));
    }
  }
}
#include "../mechsystem/verlet.hpp"
TEST_CASE("Velocity Verlet order for a linear spring") {
  struct Acceleration : NonlinearFunction {
    size_t dimX() const override{return 1;}
    size_t dimF() const override{return 1;}
    void evaluate(VectorView<> x,VectorView<> f) const override {f(0)=-x(0);}
    void evaluateDeriv(VectorView<>,MatrixView<double> j) const override {j(0,0)=-1;}
  };
  auto error=[](int n) {
    Vector<> q={1},v={0};
    SolveODE_Verlet(2*std::numbers::pi,n,q,v,std::make_shared<Acceleration>());
    return std::hypot(q(0)-1,v(0));
  };
  REQUIRE(std::abs(std::log2(error(64)/error(128))-2)<.1);
}
