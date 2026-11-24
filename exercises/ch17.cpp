#include "models.hpp"
#include <fstream>
#include <iomanip>
using namespace ASC_ode;

template<class Method>
void experiment(const char* name) {
  auto spring=std::make_shared<UnitSpring>();
  Method method(spring);
  const double period=2*std::numbers::pi;
  for(int n=16;n<=4096;n*=2) {
    const double h=period/n;
    Vector<> y={1,0};
    std::ofstream path(std::string(name)+"_spring_"+std::to_string(n)+".csv");
    path<<"t,p,v\n0,1,0\n"<<std::setprecision(16);
    for(int i=1;i<=n;i++) {
      method.doStep(h,y);
      path<<i*h<<','<<y(0)<<','<<y(1)<<'\n';
    }
    std::cout<<name<<','<<n<<','<<h<<','<<std::hypot(y(0)-1,y(1))<<'\n';
  }
  for(int perPeriod : {32,128}) {
    Vector<> y={1,0};
    double h=period/perPeriod;
    std::ofstream trace(std::string(name)+"_long_"+std::to_string(perPeriod)+".csv");
    trace<<"t,p,v\n0,1,0\n"<<std::setprecision(16);
    for(int i=1;i<=20*perPeriod;i++) {
      method.doStep(h,y);
      trace<<i*h<<','<<y(0)<<','<<y(1)<<'\n';
    }
  }
  for(double rc : {1.,0.0001}) {
    auto circuit=std::make_shared<Circuit>(rc);
    Method solver(circuit);
    for(int n : {100,1000,2000}) {
      Vector<> y={0,0};
      double h=0.04/n;
      std::ofstream out(std::string(name)+"_rc_"+std::to_string(rc)+"_"+std::to_string(n)+".csv");
      out<<"t,voltage,exact\n"<<std::setprecision(16);
      for(int i=1;i<=n;i++) {
        solver.doStep(h,y);
        out<<i*h<<','<<y(0)<<','<<circuit->exact(i*h)<<'\n';
      }
    }
  }
}
int main() {
  std::cout<<"method,N,h,error\n"<<std::setprecision(16);
  experiment<ExplicitEuler>("explicit"); experiment<ImplicitEuler>("implicit");
  experiment<ImprovedEuler>("midpoint"); experiment<CrankNicolson>("cn");
}
