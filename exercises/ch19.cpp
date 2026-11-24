#include "functions.hpp"
#include "explicitRK.hpp"
#include <iomanip>
using namespace ASC_ode;
void measure(const std::string& name, TimeStepper& solver) {
  for(int n=16;n<=4096;n*=2) {
    Vector<> y={1,0};
    double h=2*std::numbers::pi/n;
    for(int i=0;i<n;i++) solver.doStep(h,y);
    std::cout<<name<<','<<n<<','<<h<<','<<std::hypot(y(0)-1,y(1))<<'\n';
  }
}
int main() {
  auto spring=std::make_shared<UnitSpring>();
  std::cout<<"method,N,h,error\n"<<std::setprecision(16);
  auto rk4=classicalRK4(spring);
  measure("rk4",rk4);
  for(int s=1;s<=3;s++) {
    for(bool radau : {false,true}) {
      Vector<> c(s),weights(s);
      if(radau) GaussRadau(c,weights); else GaussLegendre(c,weights);
      auto [a,b]=computeABfromC(c);
      std::string name=(radau?"radau":"gauss")+std::to_string(s);
      for(int i=0;i<s;i++) {
        if(std::abs(b(i)-weights(i))>1e-10) throw std::runtime_error("weights disagree");
        if(!radau) {
          double x=std::cos(std::numbers::pi*(s-1-i+0.75)/(s+.5));
          for(int iter=0;iter<20;iter++) {
            auto p=legendre(s,AutoDiff<1>(Variable<0>(x)));
            x-=p.value()/p.deriv()[0];
            if(std::abs(p.value())<1e-15) break;
          }
          if(std::abs((x+1)/2-c(i))>1e-13) throw std::runtime_error("nodes disagree");
        }
        std::cerr<<name<<" c="<<c(i)<<" A=";
        for(int j=0;j<s;j++) std::cerr<<' '<<a(i,j);
        std::cerr<<" b="<<b(i)<<'\n';
      }
      ImplicitRungeKutta method(spring,a,b,c);
      measure(name,method);
    }
  }
}
