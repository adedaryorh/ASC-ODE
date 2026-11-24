#include "adaptive.hpp"
#include "models.hpp"
#include <fstream>
#include <iomanip>
using namespace ASC_ode;
// The explicit orbit stepper only calls evaluate.
class Orbit : public NonlinearFunction {
public:
  size_t dimX() const override {return 4;}
  size_t dimF() const override {return 4;}
  void evaluate(VectorView<> y,VectorView<> f) const override {
    constexpr double mu=.012277471;
    double x=y(0),z=y(1),u=y(2),v=y(3);
    double r1=std::pow((x+mu)*(x+mu)+z*z,1.5);
    double r2=std::pow((x-1+mu)*(x-1+mu)+z*z,1.5);
    f(0)=u; f(1)=v;
    f(2)=x+2*v-(1-mu)*(x+mu)/r1-mu*(x-1+mu)/r2;
    f(3)=z-2*u-(1-mu)*z/r1-mu*z/r2;
  }
  void evaluateDeriv(VectorView<> y,MatrixView<double> j) const override {
    constexpr double mu=.012277471;
    j=0.; j(0,2)=j(1,3)=1; j(2,0)=j(3,1)=1;
    j(2,3)=2; j(3,2)=-2;
    for(auto [dx,m] : {std::pair{y(0)+mu,1-mu},std::pair{y(0)-1+mu,mu}}) {
      double z=y(1),r=std::hypot(dx,z),r3=std::pow(r,3),r5=r3*r*r;
      j(2,0)+=m*(3*dx*dx/r5-1/r3); j(3,1)+=m*(3*z*z/r5-1/r3);
      j(2,1)+=3*m*dx*z/r5; j(3,0)+=3*m*dx*z/r5;
    }
  }
};
class Oregonator : public NonlinearFunction {
  static constexpr double a=1.34,b=1.6e9,c=8e3,d=4e7;
public:
  size_t dimX() const override{return 5;}
  size_t dimF() const override{return 5;}
  void evaluate(VectorView<> y,VectorView<> f) const override {
    double r1=a*y(0)*y(1),r2=b*y(1)*y(2),r3=c*y(0)*y(2),r4=d*y(2)*y(2),r5=y(4);
    f(0)=-r1-r3+r4; f(1)=-r1-r2+r5; f(2)=r1-r2+r3-2*r4;
    f(3)=r1+2*r2+r4; f(4)=r3-r5;
  }
  void evaluateDeriv(VectorView<> y,MatrixView<double> j) const override {
    double A=a*y(1),B=a*y(0),C=b*y(2),D=b*y(1),E=c*y(2),F=c*y(0),G=2*d*y(2);
    j=0.;
    j(0,0)=-A-E; j(0,1)=-B; j(0,2)=-F+G;
    j(1,0)=-A; j(1,1)=-B-C; j(1,2)=-D; j(1,4)=1;
    j(2,0)=A+E; j(2,1)=B-C; j(2,2)=-D+F-2*G;
    j(3,0)=A; j(3,1)=B+2*C; j(3,2)=2*D+G;
    j(4,0)=E; j(4,2)=F; j(4,4)=-1;
  }
};
int main() {
  auto spring=std::make_shared<UnitSpring>();
  BogackiShampine bs(spring);
  for(double tol : {1e-5,1e-6,1e-7,1e-8}) {
    Vector<> y={1,0};
    auto counts=SolveODE_Adaptive(bs,2*std::numbers::pi,.1,tol,0,y);
    std::cout<<"tol "<<tol<<" error "<<std::hypot(y(0)-1,y(1))<<" steps "<<counts.accepted<<'\n';
  }
  auto orbit=std::make_shared<Orbit>();
  BogackiShampine adaptive(orbit);
  const double period=17.0652166;
  Vector<> initial={.994,0,0,-2.0015851063790825};
  auto distance=[&](VectorView<> y){return 384000*std::hypot(y(0)-initial(0),y(1));};
  for(double tol : {1e-8,1e-9,1e-10,1e-11,1e-12}) {
    Vector<> y=initial;
    std::ofstream out("orbit.csv"); out<<"t,h,x,y\n0,0,.994,0\n"<<std::setprecision(16);
    auto counts=SolveODE_Adaptive(adaptive,period,.001,tol,tol,y,[&](double t,double h,VectorView<> state){
      out<<t<<','<<h<<','<<state(0)<<','<<state(1)<<'\n';
    });
    std::cout<<"orbit tol "<<tol<<" km "<<distance(y)<<" accepted "<<counts.accepted<<" rejected "<<counts.rejected<<'\n';
    if(distance(y)<=1) break;
  }
  auto rk=classicalRK4(orbit);
  for(int n=1024;n<=1048576;n*=2) {
    Vector<> y=initial;
    for(int i=0;i<n;i++) rk.doStep(period/n,y);
    std::cout<<"orbit RK4 "<<n<<" km "<<distance(y)<<'\n';
    if(distance(y)<=1) break;
  }
  auto chem=std::make_shared<Oregonator>();
  ImplicitEuler ie(chem); Vector<> nodes={1./3,1};
  auto [a,b]=computeABfromC(nodes);
  ImplicitRungeKutta radau(chem,a,b,nodes);
  StepperPair pair(ie,radau,1); BogackiShampine explicitPair(chem);
  for(bool stiff : {true,false}) {
    Vector<> y={.05,1e-4,1e-10,.1,1e-4};
    std::ofstream out(stiff?"chem_implicit.csv":"chem_explicit.csv");
    out<<"t,c1,c2,c3,c4,c5\n0,.05,.0001,1e-10,.1,.0001\n"<<std::setprecision(16);
    double last=0;
    auto counts=SolveODE_Adaptive(stiff?static_cast<EmbeddedTimeStepper&>(pair):explicitPair,
       300,1e-4,1e-12,1e-4,y,[&](double t,double,VectorView<> values){
         if(t-last>=.02 || t==300) {
           last=t;out<<t;
           for(size_t j=0;j<5;j++) out<<','<<values(j);
           out<<'\n';
         }
       });
    std::cout<<"Oregonator "<<(stiff?"implicit":"explicit")<<" accepted "<<counts.accepted<<" rejected "<<counts.rejected<<std::endl;
  }
}
